from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

exec(open("torchextractor/version.py").read())

setup(
    name="torchextractor",  # Replace with your own username
    version=__version__,
    author="Antoine Broyelle",
    author_email="antoine.broyelle@pm.me",
    description="Pytorch feature extraction made simple",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antoinebrl/torchextractor",
    project_urls={
        "Bug Tracker": "https://github.com/antoinebrl/torchextractor/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="pytorch torch feature extraction",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6",
    install_requires=["numpy", "torch>=1.4.0"],
)
