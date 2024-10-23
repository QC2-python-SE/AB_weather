from setuptools import setup, find_packages

setup(
    name="abweather",  # Replace with your package name
    version="0.1",  # Replace with your package version
    author="Abbie Bray and Bruno Camino",  # Your name or organization
    author_email="quantumpython@ucl.ac.uk",  # Your contact email
    description="A python package for weather prediction",  # Short description of your package
    long_description=open('README.md').read(),  # Long description read from README file
    long_description_content_type="text/markdown",  # Content type of the long description
    url="https://github.com/QC2-python-SE/AB_weather",  # URL of the project, if available
    packages=find_packages(),  # Automatically find and include all packages in your project
    install_requires=[
        # List your package dependencies here
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "pandas"
        # Add more dependencies as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",  # Minimum Python version
        "License :: OSI Approved :: MIT License",  # License information
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum Python version required
    license="MIT",  # License type
    keywords="machine-learning weather",  # Keywords relevant to your package
)