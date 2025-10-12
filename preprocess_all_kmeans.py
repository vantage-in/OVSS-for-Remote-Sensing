import os

# 임베딩이 저장된 기본 디렉토리
BASE_EMBEDDING_DIR = './embeddings'
# K-means 결과가 저장될 기본 디렉토리
BASE_OUTPUT_DIR = './kmeans'

# --- [추가] 여러 Seed에 대한 실험을 위한 리스트 ---
DATASETS_TO_RUN = ['udd5', 'vdd'] 
SEEDS_TO_RUN = [486, 2025] # 원하는 만큼 seed를 추가하세요

if not os.path.isdir(BASE_EMBEDDING_DIR):
    print(f"Error: Base embedding directory not found at '{BASE_EMBEDDING_DIR}'")
    exit()

if DATASETS_TO_RUN:
    # 지정된 데이터셋만 사용
    dataset_dirs = DATASETS_TO_RUN
    print(f"✅ Running preprocessing for selected datasets: {dataset_dirs}")
else:
    # 모든 하위 디렉토리를 탐색
    dataset_dirs = [d for d in os.listdir(BASE_EMBEDDING_DIR) if os.path.isdir(os.path.join(BASE_EMBEDDING_DIR, d))]
    print("✅ Running preprocessing for all available datasets.")

# Seed 루프를 바깥쪽에 두어 seed별로 모든 데이터셋을 처리
for seed in SEEDS_TO_RUN:
    print("#"*80)
    print(f" S T A R T I N G   P R O C E S S   F O R   S E E D : {seed} ")
    print("#"*80)
    
    for dataset_name in dataset_dirs:
        embedding_dir = os.path.join(BASE_EMBEDDING_DIR, dataset_name)
        
        # --- [수정] Seed를 포함한 출력 디렉토리 경로 생성 ---
        output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
        
        command = (
            f"python preprocess_kmeans.py "
            f"--embedding-dir {embedding_dir} "
            f"--output-dir {output_dir} "
            f"--seed {seed}" # Seed 인자 전달
        )
        
        print("="*80)
        print(f"🚀 Starting K-Means preprocessing for dataset: [{dataset_name.upper()}] with Seed: [{seed}]")
        print(f"   - Input:   {embedding_dir}")
        print(f"   - Output:  {output_dir}")
        print("="*80)
        
        os.system(command)

print("\n🎉 All K-Means preprocessing tasks are complete for all seeds!")