from setuptools import setup, find_packages

setup(
    name="movie_genre_classifier",
    version="0.1.0",
    description="A TF-IDF + ML pipeline to classify movie plots by genre",
    author="Bekam Guta", # Or your actual name/organization
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # you can either list your main deps here,
        # or leave this empty if you’ll rely on requirements.txt
        "pandas",
        "scikit-learn",
        "spacy",
        "joblib",
        # etc…
    ],
    python_requires=">=3.8",
)