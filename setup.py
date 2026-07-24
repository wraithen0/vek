from setuptools import setup, Extension
import numpy as np

setup(
    name="vek",
    version="0.8.0",
    ext_modules=[
        Extension(
            "_vek_cext",
            sources=["src/python/vek_cext.c"],
            include_dirs=["include", np.get_include()],
            library_dirs=["."],
            libraries=["vek", "pthread", "m"],
            extra_compile_args=["-O3", "-march=native"],
        )
    ],
)
