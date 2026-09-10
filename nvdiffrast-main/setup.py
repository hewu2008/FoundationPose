import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nvdiffrast',
    packages=['nvdiffrast'],
    package_dir={'nvdiffrast': 'nvdiffrast'},
    ext_modules=[
        CUDAExtension(
            name='_nvdiffrast_c',
            sources=[
                'csrc/common/common.cpp',
                'csrc/common/texture.cpp',
                'csrc/common/cudaraster/impl/Buffer.cpp',
                'csrc/common/cudaraster/impl/CudaRaster.cpp',
                'csrc/common/cudaraster/impl/RasterImpl.cpp',
                'csrc/torch/torch_bindings.cpp',
                'csrc/torch/torch_antialias.cpp',
                'csrc/torch/torch_interpolate.cpp',
                'csrc/torch/torch_rasterize.cpp',
                'csrc/torch/torch_texture.cpp',
                'csrc/common/antialias.cu',
                'csrc/common/interpolate.cu',
                'csrc/common/rasterize.cu',
                'csrc/common/texture_kernel.cu',
                'csrc/common/cudaraster/impl/RasterImpl_kernel.cu',
            ],
            include_dirs=['csrc', 'csrc/common', 'csrc/torch', 'csrc/common/cudaraster/impl'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-fPIC'],
                'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
)
