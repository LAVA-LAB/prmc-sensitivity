import os
import sys

from setuptools import setup
from setuptools.command.test import test

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


class PyTest(test):
    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(['test'])
        sys.exit(errno)


setup(
    name='prmcDifferentiator',
    version='0.1',
    author='T. Badings',
    author_email='thom.badings@ru.nl',
    description='Differentiating solution functions for parametric (robust) Markov chains',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['prmcDifferentiator'],
    cmdclass={
        'test': PyTest
    },
    zip_safe=False,
    install_requires=[
        'stormpy>=1.7.0', # Back-end model checker
        'cvxpy>=1.2.2',
        'gurobipy>=10.0.0'
        'numpy>=1.23.5',
        'tqdm>=4.64.1', # Progress bar
        'pandas>=1.5.2',
        'setuptools>=45.2.0',
        'scipy>=1.9.3', # For sparse matrices
        'tabulate>=0.9.0',
        'skikit-umfpack'>='0.3.1' # Speed up SciPy
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    python_requires='>=3',
)
