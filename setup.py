# building ML application as a package
from setuptools import find_packages, setup
from typing import List
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str)->List[str]:
    '''
    This will return llst of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        print(requirements)
    return requirements


setup(
    name='circleDetection',
    version= '0.0.1',
    author= 'Vaishnavi',
    author_email='vaishnavi.bhargava2605@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt'),
)
