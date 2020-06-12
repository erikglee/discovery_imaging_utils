import setuptools

setup(name='eriks_packages',..)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="discovery_imaging_utils", # Replace with your own username
    version="0.0.1",
    author="Erik Lee",
    author_email="leex6144@umn.edu",
    description="A package to aid in resting-state fMRI analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erikglee/discovery_imaging_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
