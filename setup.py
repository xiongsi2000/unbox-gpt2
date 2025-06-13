from setuptools import setup, find_packages

setup(
    name="transformer-educational",
    version="1.0.0",
    description="Educational implementation of Transformer architecture",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0", 
        "regex>=2021.0.0"
    ],
    python_requires=">=3.8",
    author="Educational Project",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)