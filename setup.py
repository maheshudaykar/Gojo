"""Setup script for Gojo package."""
from pathlib import Path
from setuptools import setup, find_packages

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="gojo",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Gojo: production-grade phishing URL detector with ML ensemble and Thompson Sampling RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gojo",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gojo/issues",
        "Documentation": "https://github.com/yourusername/gojo#readme",
        "Source Code": "https://github.com/yourusername/gojo",
    },
    packages=find_packages(exclude=["tests", "tests.*", "data", "logs", "models"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Typing :: Typed",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "production": [
            "waitress>=3.0.0",
            "gunicorn>=21.2.0; platform_system!='Windows'",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "pylint>=2.17.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gojo=phish_detector.__main__:main",
            "gojo-train=phish_detector.train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "phish_detector": ["py.typed"],
        "webapp": ["templates/*.html", "static/*"],
    },
    zip_safe=False,
    keywords=[
        "phishing",
        "security",
        "machine-learning",
        "reinforcement-learning",
        "thompson-sampling",
        "url-analysis",
        "cybersecurity",
        "fraud-detection",
    ],
)
