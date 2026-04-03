"""
Сборка Cython: pip install -e .

На Windows нужен MSVC Build Tools. Если сборка не удалась — используются чистые Python-фолбэки.
"""
from pathlib import Path

import numpy as np
from setuptools import Extension, find_packages, setup

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

ROOT = Path(__file__).resolve().parent
NP_INC = np.get_include()

extensions = [
    Extension(
        name="src.fast._fractals",
        sources=[str(ROOT / "src" / "fast" / "_fractals.pyx")],
        include_dirs=[NP_INC],
    ),
    Extension(
        name="src.fast._imbalance",
        sources=[str(ROOT / "src" / "fast" / "_imbalance.pyx")],
        include_dirs=[NP_INC],
    ),
    Extension(
        name="src.fast._footprint",
        sources=[str(ROOT / "src" / "fast" / "_footprint.pyx")],
        include_dirs=[NP_INC],
    ),
]

setup(
    name="ferrari-liquidity",
    version="0.2.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}) if cythonize else [],
    zip_safe=False,
)
