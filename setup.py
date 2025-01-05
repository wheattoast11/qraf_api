from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qraf",
    version="0.1.0",
    author="QRAF Team",
    author_email="team@qraf.ai",
    description="Quantum Reasoning Augmentation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qraf-team/qraf",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.3.0",
            "flake8>=4.0.1",
            "mypy>=0.950",
            "isort>=5.10.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "qraf=qraf.cli:run",
        ],
    },
) 