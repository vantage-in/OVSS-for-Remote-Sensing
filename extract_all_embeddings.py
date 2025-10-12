import os
import re

# 여러 데이터셋의 설정 파일 리스트
# 주석이나 빈 줄은 자동으로 무시됩니다.
configs_list_str = """
# rs semantic segmentation
# ./configs/cfg_openearthmap.py
./configs/cfg_loveda.py
# ./configs/cfg_iSAID.py
./configs/cfg_potsdam.py
./configs/cfg_vaihingen.py
./configs/cfg_uavid.py
./configs/cfg_udd5.py
# ./configs/cfg_vdd.py
# rs single-class
./configs/cfg_whu_aerial.py
./configs/cfg_whu_sat_II.py
./configs/cfg_inria.py
./configs/cfg_xBD.py
./configs/cfg_chn6-cug.py
./configs/cfg_deepglobe_road.py
./configs/cfg_massachusetts_road.py
./configs/cfg_spacenet_road.py
./configs/cfg_wbs-si.py
"""

# 문자열을 파싱하여 실제 설정 파일 경로 리스트 생성
configs_list = [line.strip() for line in configs_list_str.strip().split('\n') if line.strip() and not line.strip().startswith('#')]

# 모든 임베딩이 저장될 기본 디렉토리
BASE_OUTPUT_DIR = './embeddings'
# 임시 작업 로그가 저장될 디렉토리
WORK_DIR = './work_tmp/embedding_extraction/'

# 리스트의 모든 설정 파일에 대해 순차적으로 임베딩 추출 실행
for config_path in configs_list:
    if not os.path.exists(config_path):
        print(f"⚠️ Warning: Config file not found, skipping: {config_path}")
        continue

    # 설정 파일 이름에서 데이터셋 이름 추출 (예: cfg_loveda.py -> loveda)
    basename = os.path.basename(config_path)
    dataset_name = os.path.splitext(basename)[0].replace('cfg_', '')
    
    # 데이터셋별로 독립된 출력 폴더 경로 생성
    output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)

    # 실행할 명령어 구성
    # 멀티 GPU 환경을 사용하신다면 'python' 부분을 'torchrun --nproc_per_node=4' 등으로 변경할 수 있습니다.
    command = (
        f"python extract_embeddings.py {config_path} "
        f"--output-dir {output_dir} "
        f"--work-dir {WORK_DIR}/{dataset_name}"
    )
    
    # 실행 전, 어떤 작업이 수행되는지 터미널에 출력
    print("="*80)
    print(f"🚀 Starting embedding extraction for dataset: [{dataset_name.upper()}]")
    print(f"   - Config: {config_path}")
    print(f"   - Output: {output_dir}")
    print(f"   - Command: {command}")
    print("="*80)
    
    # 명령어 실행
    os.system(command)

print("\n🎉 All embedding extraction processes are complete!")