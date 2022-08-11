import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

base_reqs = [
        "openmdao == 3.15.0",
        "ruamel.yaml >= 0.16",
        "plasmapy >= 0.5.0",
        "numpy >= 1.21.0",
]

docs_reqs = [
        "docutils<0.17,>=0.15",
        "nbclient<0.6,>=0.2",
        "sphinx >= 4.4.0",
        "jupyter-sphinx==0.3.2",
        "Jinja2<3.1",
        "sphinxcontrib-bibtex >= 2.3",
        "jsonschema < 4",
        "sphinx_rtd_theme",
]

setuptools.setup(
    name="faroes",
    version="0.0.1",
    author="Jacob Schwartz",
    author_email="jacobas@princeton.edu",
    description="Analyze tokamak designs for techno-economic analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cfe316/FAROES",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=base_reqs,
    extras_require={
        "docs": docs_reqs
    },
)
