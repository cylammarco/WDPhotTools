from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'numpy>=1.16', 'scipy>=1.5', 'matplotlib>3.0', 'emcee>=3.0', 'corner'
]

__packagename__ = "WDLFBuilder"

setup(name=__packagename__,
      version='0.1.0',
      packages=find_packages(),
      author='Marco Lam',
      author_email='cylammarco@gmail.com',
      description="WDPhotTools",
      url="https://github.com/cylammarco/WDPhotTools",
      license='bsd-3-clause',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      zip_safe=False,
      include_package_data=True,
      install_requires=install_requires,
      python_requires='>=3.6')
