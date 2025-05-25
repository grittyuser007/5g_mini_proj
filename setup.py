from setuptools import setup, find_packages

setup(
    name="injury_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0.74",
        "numpy>=1.24.3",
        "tensorflow>=2.13.0",
        "keras>=2.13.1",
        "pillow>=10.0.0",
        "matplotlib>=3.7.2",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.3",
        "firebase-admin>=6.2.0",
        "google-cloud-storage>=2.10.0",
        "deepface>=0.0.79",
        "pyyaml>=6.0.1",
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "ultralytics>=8.0.145",
    ],
    entry_points={
        'console_scripts': [
            'injury-detector=src.main:main',
        ],
    },
)
