import setuptools

with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="pyvdw",
    version="v0.1.0",
    author="ajz34",
    author_email="ajz34@outlook.com",
    description="My wrapper of various existing vDW libraries for PySCF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajz34/vdw",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pyscf"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
)
