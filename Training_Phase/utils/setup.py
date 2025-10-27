from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

package = Extension(
    'compute_overlap',
    sources=['compute_overlap.pyx'],
    include_dirs=[numpy.get_include()]
)

setup(ext_modules=cythonize([package]))
