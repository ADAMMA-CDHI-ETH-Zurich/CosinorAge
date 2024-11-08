from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='cosinorage',
    version='0.1.3',
    description='A package for computing the ConsinorAge from raw '
                'accelerometer data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jacob Leo Oskar Hunecke',
    url='https://github.com/jlohunecke/CosinorAge.git',
    packages=find_packages(),  # Automatically finds subpackages
    install_requires=[],  # Add any dependencies here

    extras_require={
        'docs': [
            'sphinx',
            'furo',
            'pandas',
            # other doc dependencies
        ],
    },
)
