from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cass-system",
    version="0.1.0",
    author="katoki-dev",
    description="Campus AI Safety & Surveillance System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "transformers>=4.30.0",
    ],
    entry_points={
        "console_scripts": [
            "cass-server=cass.api.main:run",
            "cass-train=cass.training.train:main",
            "cass-test=cass.testing.test_models:main",
            "cass-infer=cass.inference.pipeline:main",
        ],
    },
)
