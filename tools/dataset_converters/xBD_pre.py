import os
import glob
import shutil
import argparse
from tqdm import tqdm

def filter_and_copy_files(base_dir, source_folder, dest_folder, keyword):
    """
    지정된 폴더에서 키워드가 포함된 파일을 찾아 새 폴더로 복사합니다.

    Args:
        base_dir (str): 데이터셋의 최상위 경로.
        source_folder (str): 원본 파일들이 있는 폴더 이름 (예: 'images').
        dest_folder (str): 복사할 대상 폴더 이름 (예: 'images_pre').
        keyword (str): 파일명에서 찾을 키워드 (예: '_pre_disaster').
    """
    # 원본 폴더와 대상 폴더의 전체 경로 설정
    source_path = os.path.join(base_dir, source_folder)
    dest_path = os.path.join(base_dir, dest_folder)

    # 경로 존재 여부 확인
    if not os.path.isdir(source_path):
        print(f"오류: 원본 폴더를 찾을 수 없습니다: {source_path}")
        return

    # 대상 폴더 생성
    os.makedirs(dest_path, exist_ok=True)
    print(f"'{dest_path}' 폴더를 확인/생성했습니다.")

    # 원본 폴더 내의 모든 png 파일 검색
    search_pattern = os.path.join(source_path, '*.png')
    file_list = glob.glob(search_pattern)

    # 키워드가 포함된 파일만 필터링
    pre_disaster_files = [f for f in file_list if keyword in os.path.basename(f)]

    if not pre_disaster_files:
        print(f"경고: '{source_path}'에서 '{keyword}' 키워드를 포함한 파일을 찾지 못했습니다.")
        return

    print(f"'{source_folder}' 폴더에서 {len(pre_disaster_files)}개의 'pre' 이미지를 찾아 복사를 시작합니다.")

    # 파일 복사 진행
    for src_file in tqdm(pre_disaster_files, desc=f"Copying to {dest_folder}"):
        dest_file = os.path.join(dest_path, os.path.basename(src_file))
        shutil.copy(src_file, dest_file)

def main():
    parser = argparse.ArgumentParser(description="'pre_disaster' 이미지만 선별하여 새 폴더에 복사하는 스크립트")
    parser.add_argument('dataset_dir', type=str, help='데이터셋의 최상위 디렉토리 경로')
    
    args = parser.parse_args()
    
    keyword_to_find = '_pre_disaster'

    # 'images' 폴더 처리
    filter_and_copy_files(args.dataset_dir, 'images', 'images_pre', keyword_to_find)
    
    # 'targets_cvt' 폴더 처리
    filter_and_copy_files(args.dataset_dir, 'targets_cvt', 'targets_cvt_pre', keyword_to_find)
    
    print("\n모든 작업이 완료되었습니다.")

if __name__ == '__main__':
    main()