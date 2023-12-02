from setuptools import find_packages, setup

def get_requirements(file_path):
    requirements_list = []
    with open(file_path) as file_path:
        for file in file_path.readlines():
            requirements_list.append(file.replace("\n", ''))

    requirements_list.remove('-e .')
    return requirements_list

setup(
    name         = 'machine_learning_project',
    version      = '0.0.2',
    author       = 'Firstname Lastname',
    author_email = 'abc@email.com',
    packages     = find_packages(),
    install_requires = get_requirements('requirements.txt'),

)