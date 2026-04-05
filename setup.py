from setuptools import setup, find_packages

setup(
    name="movie_genre_classifier",
    version="0.1.0",
    description="A TF-IDF + ML pipeline to classify movie plots by genre",
    author="Bekam Guta",
    url="https://github.com/bekamguta/movie-genre-classifier",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "pandas>=2.3.0",
        "scikit-learn>=1.7.0",
        "spacy>=3.8.7",
        "joblib>=1.5.1",
        "matplotlib>=3.10.3",
        "numpy>=2.3.1",
    ],
    extras_require={
        "dev": ["pytest>=8.4.1", "seaborn>=0.13.2"],
    },
    entry_points={
        "console_scripts": [
            "mgc-train=src.main:main",
            "mgc-predict=src.models.predict:main",
        ]
    },
    python_requires=">=3.10",
)
