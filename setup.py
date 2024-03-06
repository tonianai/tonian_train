from setuptools import find_packages, setup

setup(

    name='tonian_train',
    packages=find_packages(),
    version='0.3.1',
    description='The training library used to train the simulated tonian robots',
    author='Jonas Schmidt',
    install_requires=['torch', 'numpy', 'gym', 'tensorboard', 'pyyaml'],
    
)
