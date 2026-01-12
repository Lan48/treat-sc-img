import torch
import pynvml

def get_least_used_gpu():
    """获取当前内存使用最少的GPU设备"""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        # 获取每个GPU的剩余内存
        gpu_mem = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem.append(info.free)
        
        # 选择剩余内存最多的GPU
        best_gpu = torch.device(f"cuda:{gpu_mem.index(max(gpu_mem))}")
        print(f"Selected GPU {best_gpu.index} with {max(gpu_mem)/1024**2:.2f} MB free memory")
        return best_gpu
    except Exception as e:
        print(f"Error selecting GPU: {e}. Using default GPU")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
