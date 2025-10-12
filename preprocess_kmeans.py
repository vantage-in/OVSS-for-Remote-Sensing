'''
python preprocess_kmeans.py --embedding-dir ./embeddings/vdd --output-dir ./kmeans/vdd/ --seed 42
'''

import argparse
import os
import torch
from tqdm import tqdm
import glob
from fast_pytorch_kmeans import KMeans  # fast_pytorch_kmeans 라이브러리가 필요합니다.
# from torch_kmeans import KMeans

def parse_args():
    """스크립트 실행을 위한 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Pre-process K-Means sampling for context features')
    parser.add_argument('--embedding-dir', required=True, help='Directory containing the extracted embeddings')
    parser.add_argument('--output-dir', required=True, help='Directory to save the K-Means results')
    parser.add_argument('--device', default='cuda:0', help='Device to run K-Means on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for K-Means and sampling')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    final_output_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(final_output_dir, exist_ok=True)
    embedding_files = glob.glob(os.path.join(args.embedding_dir, '*.pt'))
    
    if not embedding_files:
        print(f"⚠️ No embedding files found in {args.embedding_dir}. Skipping.")
        return

    print(f"Processing {len(embedding_files)} images from {args.embedding_dir} with seed {args.seed}...")

    for file_path in tqdm(embedding_files, desc=f"Seed {args.seed}"):
        try:
            embeddings = torch.load(file_path, map_location=device)
            # ... (이전과 동일한 임베딩 로드 코드)
            all_robust_dino_feats = embeddings['all_robust_dino_feats']
            all_robust_clip_feats = embeddings['all_robust_clip_feats']
            all_robust_patch_ids = embeddings['all_robust_patch_ids']
        except Exception as e:
            print(f"Could not load or process {file_path}. Error: {e}")
            continue

        num_patches = int(torch.max(all_robust_patch_ids)) + 1
        all_sampled_features = {}

        for patch_counter in range(num_patches):
            # ... (external_mask, external_dino_feats 등 이전과 동일)
            external_mask = (all_robust_patch_ids != patch_counter)
            external_dino_feats = all_robust_dino_feats[external_mask]
            external_clip_feats = all_robust_clip_feats[external_mask]
            num_external_robust = external_dino_feats.shape[0]

            sampled_dino, sampled_clip = None, None

            if num_external_robust > 0:
                K = 25
                K = min(K, num_external_robust)
                
                # --- [수정] K-means 및 샘플링에 Seed 적용 ---
                torch.manual_seed(args.seed) # K-means 초기화에 seed 적용
                kmeans = KMeans(n_clusters=K, init_method="kmeans++", mode="cosine", max_iter=25, verbose=False)
                labels = kmeans.fit_predict(external_dino_feats)
                # kmeans = KMeans(n_clusters=K, init_method='k-means++', verbose=False, seed=args.seed)
                # labels = kmeans.fit_predict(external_dino_feats.unsqueeze(0)).squeeze(0)

                M = 80
                dino_samples, clip_samples = [], []
                
                for cid in range(K):
                    member_indices = (labels == cid).nonzero(as_tuple=True)[0]
                    if len(member_indices) == 0: continue
                    
                    torch.manual_seed(args.seed) # 샘플링 과정에도 동일한 seed 적용
                    # ... (이후 샘플링 로직은 동일)
                    if len(member_indices) > M:
                        rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
                    else:
                        rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]
                    
                    dino_samples.append(external_dino_feats[rand_indices])
                    clip_samples.append(external_clip_feats[rand_indices])

                if dino_samples:
                    sampled_dino = torch.cat(dino_samples, dim=0)
                    sampled_clip = torch.cat(clip_samples, dim=0)
            
            patch_results = {
                'sampled_dino': sampled_dino.cpu() if sampled_dino is not None else None,
                'sampled_clip': sampled_clip.cpu() if sampled_clip is not None else None,
            }
            all_sampled_features[patch_counter] = patch_results

        basename = os.path.basename(file_path)
        output_path = os.path.join(final_output_dir, basename)
        torch.save(all_sampled_features, output_path)

if __name__ == '__main__':
    main()