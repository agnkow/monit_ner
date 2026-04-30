from setuptools import setup

setup(
    name='monitner',
    version='0.1',
    description='Python package for monitoring a NER model (spaCy, Polish)',
    author='Agnieszka Kowalik',
    author_email='',
    include_package_data=False,
    packages=['monitner'],
    install_requires=[
        'numpy>=2.4.2',
        'pandas>=2.3.3',
        'scipy>=1.17.0',
        'sklearn>=1.8.0'
        'spacy>=3.8.11',
        'tqdm>=4.67.3',
    ]
)
