"""vek - Vector Engine Kernels
Builds the Python C extension. Requires libvek.so to be built first (run `make lib`).
"""
from setuptools import setup, Extension
import subprocess
import sys

def build_library():
    """Build the C library if not already built."""
    subprocess.run(["make", "lib"], check=True)

# Ensure library is built before extension
build_library()

import numpy as np

setup(
    name="vek",
    version="1.0.0",
    ext_modules=[
        Extension(
            "_vek_cext",
            sources=["src/python/vek_cext.c"],
            include_dirs=["include", np.get_include()],
            library_dirs=["."],
            libraries=["vek", "pthread", "m"],
            extra_compile_args=["-O3"],
        )
    ],
)
