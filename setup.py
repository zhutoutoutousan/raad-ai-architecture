from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="architectural-style-classifier",
    version="1.0.0",
    author="Anonymous Researchers",
    author_email="anonymous@institution.edu",
    description="EfficientNet-B0 based architectural style classification model achieving 99.7% accuracy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/architectural-style-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "albumentations>=1.3.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.62.0",
        "transformers>=4.20.0",
        "kagglehub>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.ckpt"],
    },
    entry_points={
        "console_scripts": [
            "architectural-classifier=src.cli:main",
        ],
    },
    keywords=[
        "computer-vision",
        "deep-learning",
        "architectural-classification",
        "efficientnet",
        "transfer-learning",
        "heritage-preservation",
        "image-recognition",
        "pytorch",
        "lightning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/anonymous/architectural-style-classifier/issues",
        "Source": "https://github.com/anonymous/architectural-style-classifier",
        "Documentation": "https://github.com/anonymous/architectural-style-classifier#readme",
        "Paper": "https://arxiv.org/abs/XXXX.XXXXX",
    },
)
