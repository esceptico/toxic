from pathlib import Path

from setuptools import find_packages, setup


THIS_DIR = Path(__file__).parent


def read_requirements(path_dir):
    with open(path_dir / 'requirements.txt') as file:
        lines = [ln.strip() for ln in file.readlines()]
    return lines


setup(
    name='toxic',
    version='0.1.0',
    packages=find_packages(exclude=['tests', 'tests.*']),
    author='Timur Ganiev',
    author_email='ganiev.tmr@gmail.com',
    description='Toxic comment classification and interpretation'
)
