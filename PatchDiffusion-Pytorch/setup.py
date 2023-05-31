from setuptools import setup

setup(
    name="patch-diffusion",
    py_modules=["patch_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
