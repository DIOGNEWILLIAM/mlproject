from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(filepath: str) -> List[str]:
    requirements = []
    with open(filepath) as file_obj:   # correction ici
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements  # manquait un return

setup(
    name='MLPROJECT',
    version='0.0.1',
    author='william diogne',
    author_email='williamdiogne196@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')  # correction ici
)
