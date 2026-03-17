from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [Extension("pfgd_solvers", ["pfgd_solvers.pyx"])],
        language_level=3,
    ),
    include_dirs=[numpy.get_include()],
)
