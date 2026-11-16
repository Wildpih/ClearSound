from setuptools import setup, find_packages

setup(
    name="clearsound",
    version="0.1.0",
    description="Remove vocals from audio/video and intelligently restore the sound field",
    author="ClearSound Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "demucs>=4.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "simpleaudio>=1.0.4",
        "rich>=13.0.0",
        "click>=8.1.0",
        "ffmpeg-python>=0.2.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "clearsound=clearsound.main:cli",
        ],
    },
)
