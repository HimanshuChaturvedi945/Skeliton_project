from pathlib import Path
from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    requirements_path = Path(__file__).resolve().parent / file_path
    with requirements_path.open() as file_obj:
        requirements = [req.strip() for req in file_obj if req.strip()]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="mlproject",
    version="0.1.0",
    author="Himanshu",
    author_email="sujalchaturvedi1258@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("Requirement.txt"),
)