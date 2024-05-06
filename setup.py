"""TODO: Add copyright information.
"""

from setuptools import setup, find_packages
from epbd_bert.version import __version__


# add readme
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open("requirements.txt", "r") as f:
    INSTALL_REQUIRES = f.read().strip().split("\n")

setup(
    name="epbd_bert",
    version=__version__,
    author="Anowarul Kabir, Manish Bhattarai, Kim Rasmussen, Amarda Shehu, Alan Bishop, Boian S. Alexandrov, and Anny Usheva",
    author_email="akabir4@gmu.edu",
    description="Advancing Transcription Factor Binding Site Prediction Using DNA Breathing Dynamics and Sequence Transformers via Cross Attention",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    package_dir={"epbd_bert": "epbd_bert"},
    platforms=["Linux", "Mac", "Windows"],
    include_package_data=True,
    setup_requires=[
        "argparse",
        "numpy",
    ],
    url="https://github.com/lanl/EPBD-BERT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: Beta",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.11.0",
    license="License :: BSD3 License",
)
