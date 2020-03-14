from setuptools import setup, Extension
from Cython.Build import cythonize

# cython extensions build by internal setup.py files
extensions = [
    Extension(
        name = "_barni",
        sources=["extensions/math.pyx"])]

setup(
    name='BARNI',
    packages = ['barni'],
    version='0.11',
    author='Mateusz Monterial',
    author_email='mmonterial1@llnl.gov',
    url='https://',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Scientific/Engineering',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Environment :: Console'
    ],
    provides=['barni'],
    install_requires=[
        'numpy>=1.17',
        'scikit-learn>=0.20.0',
        'scipy>=1.3.0',
        'bokeh>=1.4.0',
        'pyyaml>=5.1',
        'pandas>=0.25',
        'cython>=0.2'
    ],
    setup_requires=[
        'cython>=0.2',
    ],
    test_suite='test',
    ext_modules =  cythonize(extensions),
    # include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs()
)
