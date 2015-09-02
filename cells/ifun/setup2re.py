#python setup2re.py build_ext --inplace
from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("_ifun2re", ["_ifun2re.c", "ifun2re.c"])

setup(
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
