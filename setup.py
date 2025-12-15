import os
import sys
from setuptools import setup, find_packages


if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(
    name='bhmlib',
    version='1.0.0',
    packages=find_packages(exclude=('Outputs')),
    description='Bayesian Hilbert Maps Library',
)
