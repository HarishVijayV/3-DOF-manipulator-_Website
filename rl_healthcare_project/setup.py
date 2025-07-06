from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rl-healthcare-dosage",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Driven Drug Dosage Optimization using Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rl_healthcare_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "web": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.0",
        ],
        "distributed": [
            "ray[rllib]>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-diabetes-dqn=train_diabetes_dqn:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="reinforcement-learning healthcare ai machine-learning drug-dosage diabetes",
    project_urls={
        "Bug Reports": "https://github.com/your-username/rl_healthcare_project/issues",
        "Documentation": "https://github.com/your-username/rl_healthcare_project/wiki",
        "Source": "https://github.com/your-username/rl_healthcare_project",
    },
)