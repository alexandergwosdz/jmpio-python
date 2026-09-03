from setuptools import setup, find_packages

setup(
    name="jmpio",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
    ],
    python_requires=">=3.10",
)
