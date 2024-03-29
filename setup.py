from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    

setup(
    name='geo-helper',
    version='0.0.1',
    description="Helper package for PIER project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/icedumpy/geo-helper",
    author='Pongporamat',
    author_email='pongporamat.c@gmail.com',
    packages=find_packages(),    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
    python_requires='>=3.6',
)
