from setuptools import setup, find_packages

setup(
    name="fl_toolkit",
    version="0.1.0",
    packages=find_packages(),
    
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    
    author="Adam Piaseczny",
    author_email="apiasecz@purdue.edu",
    description="Basic toolkit for Federated Learning Research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="federated-learning, distributed-deep-learning, distributed-machine-learning",
    
    url="https://github.com/Adampi210/Federated_Learning_Base_Toolkit_torch.git",
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    python_requires='>=3.8',
)