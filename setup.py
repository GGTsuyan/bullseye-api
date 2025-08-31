from setuptools import setup, find_packages

setup(
    name="bullseye-api",
    version="1.0.0",
    description="Bullseye Dart Detection API",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "tensorflow-cpu==2.20.0",
        "opencv-python-headless",
        "numpy==1.26.0",
        "psutil",
        "Pillow",
        "typing-extensions<4.6.0",
    ],
    python_requires=">=3.9,<3.12",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
