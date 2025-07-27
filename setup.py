from setuptools import setup, Extension, find_packages
import os
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "OPT4TorchDataSet.cachelib",
        sources=["./src/OPT4TorchDataSet/cachelib.py"],
    ),
]

setup(
    ext_modules=cythonize(ext_modules, nthreads=os.cpu_count() or 1, annotate=True),
)
