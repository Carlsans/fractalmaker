from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

sfc_module = Extension(
    'superfastcode',
    sources=['module2.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='superfastcode',
    version='1.3',
    description='Python package with superfastcode2 C++ extension (PyBind11)',
    ext_modules=[sfc_module],
)