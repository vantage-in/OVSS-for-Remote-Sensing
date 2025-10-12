import torch
import numpy as np

def check_tensor_scale(saved_tensor_path: str):
    """
    .pt 파일에 저장된 '__debug_inputs' 텐서를 로드하여 값의 범위를 확인합니다.
    
    :param saved_tensor_path: 확인할 .pt 파일의 경로
    """
    print(f"--- Checking Tensor Scale for '{saved_tensor_path}' ---")
    
    # 1. .pt 파일 로드
    try:
        saved_data = torch.load(saved_tensor_path)
        # '__debug_inputs' 키로 텐서 추출
        tensor_to_check = saved_data['__debug_inputs']
        print(f"✅ Tensor loaded successfully.")
        print(f"   - Shape: {tensor_to_check.shape}")
        print(f"   - Dtype: {tensor_to_check.dtype}")
    except FileNotFoundError:
        print(f"❌ Error: Saved tensor file not found at '{saved_tensor_path}'.")
        return
    except KeyError:
        print(f"❌ Error: '__debug_inputs' key not found in the file.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return

    # 2. 텐서의 최솟값과 최댓값 계산
    min_val = torch.min(tensor_to_check).item()
    max_val = torch.max(tensor_to_check).item()
    
    print(f"\n--- Analysis Result ---")
    print(f"🔹 Minimum value: {min_val:.4f}")
    print(f"🔹 Maximum value: {max_val:.4f}")
    
    # 3. 값의 범위에 따라 스케일 추정
    if min_val >= 0 and max_val <= 1.0:
        print("✅ Verdict: The tensor scale appears to be in the [0, 1] range.")
    elif min_val >= 0 and max_val > 1.0:
        print("✅ Verdict: The tensor scale appears to be in the [0, 255] range.")
    elif min_val < 0:
        print("✅ Verdict: The tensor contains negative values, suggesting it is already normalized.")
    else:
        print("🤔 Verdict: The tensor scale is unusual. Please check the values manually.")

if __name__ == '__main__':
    # --- 사용 예시 ---
    # .pt 파일이 저장된 실제 경로로 수정하여 사용하세요.
    try:
        path_to_saved_tensor = "embeddings/tmp_vdd/DJI_0009.pt"
        check_tensor_scale(path_to_saved_tensor)
        
    except FileNotFoundError:
        print(f"\n[INFO] Could not run the check because the example file path is not set.")
        print("Please update 'path_to_saved_tensor' with your actual file path.")