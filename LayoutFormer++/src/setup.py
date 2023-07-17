"""
pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name='layoutformer',
    version='0.0.1',
    description='LayoutFormer',
    packages=find_packages(exclude=["test_*.py"]),
    install_requires=[
    ],
)
