import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="faroes", 
    version="0.0.1",
    author="Jacob Schwartz",
    author_email="jacobas@princeton.edu",
    description="Generate and analyze tokamak designs for techno-economic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cfe316/FAROES",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=["openmdao >= 3.4",
      ],
)
