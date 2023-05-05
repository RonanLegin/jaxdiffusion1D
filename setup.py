from setuptools import setup, find_packages

setup(
    name="jaxdiffusion1d",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "jax",
        "flax",
        "numpy",
        "ml-collections",
    ],
)