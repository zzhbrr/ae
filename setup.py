import os
import torch
import torch.utils.cpp_extension as torch_cpp_ext
import cpuinfo
from setuptools import setup, find_packages, Extension

DEBUG = False

def get_compile_args():
    flags = ["-std=c++17", "-O1", "-fopenmp", "-Wno-ignored-qualifiers", "-mf16c"]
    info = cpuinfo.get_cpu_info()

    if 'avx512f' in info['flags']:
        flags.extend(["-mavx512f", "-mavx512cd", "-mavx512vl"])
    elif 'avx2' in info['flags']:
        flags.extend(["-mavx2", "-mfma"])
    
    if DEBUG:
        flags.extend(["-g"])
    return flags

def get_include_paths():
    res = torch.utils.cpp_extension.include_paths()
    # res.append("/home/zzh/codes/SpecMoE/.venv/include")
    return res

def get_library_paths():
    res = torch.utils.cpp_extension.library_paths()
    # res.extend(["/home/zzh/codes/SpecMoE/.venv/lib"])
    return res

ext_modules = []
ext_modules.append(
    Extension(
        name="specmoe._cpu_kernel",
        sources=["specmoe/csrc/flashattention.cpp"],
        include_dirs=get_include_paths(),
        library_dirs=get_library_paths(),
        libraries=['torch', 'c10', 'torch_cpu', 'torch_python', 'mkl_rt'],
        language="c++",
        extra_compile_args=get_compile_args(),
        extra_link_args=['-lpthread', '-lm', '-ldl']
    )
)

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': torch_cpp_ext.BuildExtension},
) 