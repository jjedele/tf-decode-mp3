"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution


__version__ = "0.0.1"
REQUIRED_PACKAGES = [
    "tensorflow >= 1.12.0",
]

class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


setup(
    name="tf-decode-mp3",
    version=__version__,
    description="TensorFlow Operator to decode MP3 from in-memory data.",
    author="Jeffrey Jedele",
    author_email="jeffrey.jedele@gmail.com",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    #ext_modules=[Extension('_foo', ['stub.cc'])],
    zip_safe=False,
    distclass=BinaryDistribution,
    license="Apache 2.0",
    keywords="tensorflow custom op machine learning mp3 audio",
)
