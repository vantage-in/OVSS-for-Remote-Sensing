'''
python precompute_features_all.py path/to/your/checkpoint.pth
'''
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Pre-compute features for all datasets.')
    parser.add_argument('checkpoint', help='Path to the model checkpoint file.')
    parser.add_argument('--out-root', default='./precomputed_features', help='Root directory to save all feature files.')
    args = parser.parse_args()

    # eval_all.py와 동일한 설정 파일 리스트
    configs_list = [
        # rs semantic segmentation
        './configs/cfg_openearthmap.py',
        './configs/cfg_loveda.py',
        './configs/cfg_iSAID.py',
        './configs/cfg_potsdam.py',
        './configs/cfg_vaihingen.py',
        './configs/cfg_uavid.py',
        './configs/cfg_udd5.py',
        './configs/cfg_vdd.py',
        # rs single-class
        './configs/cfg_whu_aerial.py',
        './configs/cfg_whu_sat_II.py',
        './configs/cfg_inria.py',
        './configs/cfg_xBD.py',
        './configs/cfg_chn6-cug.py',
        './configs/cfg_deepglobe_road.py',
        './configs/cfg_massachusetts_road.py',
        './configs/cfg_spacenet_road.py',
        './configs/cfg_wbs-si.py',
    ]

    # 체크포인트 파일 경로
    checkpoint_path = args.checkpoint
    
    # 특징을 저장할 최상위 디렉토리
    output_root_dir = args.out_root

    for config in configs_list:
        print("="*80)
        print(f"Starting feature pre-computation for: {config}")
        print("="*80)
        
        # 1. config 파일 경로에서 데이터셋 이름 추출
        # 예: './configs/cfg_openearthmap.py' -> 'openearthmap'
        dataset_name = os.path.basename(config).replace('cfg_', '').replace('.py', '')
        
        # 2. 데이터셋별 저장 디렉토리 경로 생성
        output_dir = os.path.join(output_root_dir, dataset_name)
        
        # 3. precompute_features.py를 실행할 명령어 생성
        command = (
            f"python precompute_features.py {config} {checkpoint_path} --out {output_dir}"
        )
        
        print(f"Running command:\n{command}\n")
        
        # 4. 명령어 실행
        os.system(command)

        print(f"\nFinished pre-computation for: {dataset_name}")
        print("-" * 80)

if __name__ == '__main__':
    main()