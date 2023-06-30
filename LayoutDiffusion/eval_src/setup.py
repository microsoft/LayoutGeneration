"""
pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name='layout_eval',
    version='0.0.1',
    description='Layout Pretraining',
    packages=find_packages(exclude=["test_*.py"]),
    install_requires=[
    ],
)
