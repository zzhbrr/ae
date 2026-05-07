import ctypes
import time
import numpy as np
import torch

# 1. 加载 CUDA runtime 库
libcudart = ctypes.CDLL('libcudart.so')
libcudart.cudaMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libcudart.cudaMallocHost.restype = ctypes.c_int

# CUDA 函数返回值错误码检查
def check_cuda_status(status):
    if status != 0:
        raise RuntimeError(f'CUDA Error: {status}')

# 2. 调用 cudaHostAlloc 分配 pinned memory
def cuda_host_alloc(size_in_bytes):
    ptr = ctypes.c_void_p()

    # method 1: cudaHostAlloc
    ## method 1 will cause error when allocate large memory(6 GB)
    # flags = 0  # cudaHostAllocDefault
    # status = libcudart.cudaHostAlloc(ctypes.byref(ptr), size_in_bytes, flags)

    # method 2: cudaMallocHost
    status = libcudart.cudaMallocHost(ctypes.byref(ptr), size_in_bytes)
    check_cuda_status(status)
    return ptr

# 3. 创建 NumPy 数组并封装 pinned memory
def make_pinned_numpy_array(ptr, size, dtype=np.float32):
    # dtype.itemsize 计算字节数
    buffer_type = ctypes.c_char * (size * np.dtype(dtype).itemsize)
    buffer = buffer_type.from_address(ptr.value)
    np_array = np.frombuffer(buffer, dtype=dtype, count=size)
    return np_array

# 4. 封装为 PyTorch tensor
def make_pinned_tensor(size_in_bytes=2.625*1024*1024*1024, dtype=torch.float16):
    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = int(size_in_bytes)
    num_elements = total_bytes // element_size

    # Step 1: Allocate pinned memory
    ptr = cuda_host_alloc(total_bytes)

    # print(ptr)

    # Step 2: Create numpy array from pinned memory
    if dtype == torch.float16:
        dtype_np = np.float16
    elif dtype == torch.float32:
        dtype_np = np.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    np_array = make_pinned_numpy_array(ptr, num_elements, dtype=dtype_np)

    # Step 3: Wrap numpy array as torch tensor
    tensor = torch.from_numpy(np_array)

    return tensor, ptr  # 返回指针是为了释放用

# 5. 手动释放 pinned memory
def free_pinned_tensor(ptr):
    status = libcudart.cudaFreeHost(ptr)
    check_cuda_status(status)

if __name__ == "__main__":
    # bytes = int(2.625*1024*1024*1024)
    # t_list = []
    # for i in range(10):
    #     t = torch.empty( (1, bytes // 2), dtype=torch.float16, device="cpu", pin_memory=True)
    #     t_list.append(t)
    #     print(f"t{i} shape: {t.shape}")
    #     print(f"t{i} is_pinned: {t.is_pinned()}")
    #     print(f"t{i} {t.numel() * t.element_size() / (1024 * 1024 * 1024):.2f} GB")
    #     print(f"t{i} {t.numel() * t.element_size()/1024/1024:.2f} MB")
    # time.sleep(1000000)
    # exit()

    tensor, pinned_ptr = make_pinned_tensor(size_in_bytes=6.25*1024*1024*1024, dtype=torch.float16)
    print("Tensor shape:", tensor.shape)
    print("Is pinned:", tensor.is_pinned())

    print(f"Pinned memory size: {tensor.numel() * tensor.element_size() / (1024 * 1024 * 1024)} GB")

    free_pinned_tensor(pinned_ptr)
    print("Pinned memory released.")

'''
ulimit -l
65536

64MB
'''