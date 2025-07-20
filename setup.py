from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="int8-quantization-tool",
    version="1.0.0",
    author="Quantization Tool Development Team",
    author_email="your.email@example.com",
    description="A comprehensive int8 quantization toolkit for PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yonggang-Xie/quant_tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "visualization": ["matplotlib>=3.3.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-tool-demo=examples.basic_quantization:main",
        ],
    },
    keywords="quantization, pytorch, int8, neural networks, deep learning, model compression",
    project_urls={
        "Bug Reports": "https://github.com/yonggang-Xie/quant_tool/issues",
        "Source": "https://github.com/yonggang-Xie/quant_tool",
        "Documentation": "https://github.com/yonggang-Xie/quant_tool/blob/main/README.md",
    },
)
