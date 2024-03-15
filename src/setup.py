from setuptools import setup, find_packages

setup(
    name='NeoCOMM',
    version='1.0',
    author='Maxime Yochum & Mariam Al Harrach',
    description='Neocortical microscale computational Model for the simulation of epileptic activity',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)