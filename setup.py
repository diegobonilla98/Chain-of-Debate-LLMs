"""
Setup script for ChainOfDebate package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Chain of Debate: A collaborative AI debate system for complex problem solving."

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="chain-of-debate",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A collaborative AI debate system for complex problem solving",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chain-of-debate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        'chain_of_debate': [
            'config/*.json',
            'config/*.yaml',
        ],
    },
    entry_points={
        'console_scripts': [
            'chain-of-debate=chain_of_debate.cli:main',
        ],
    },
    keywords="ai debate llm reasoning problem-solving",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/chain-of-debate/issues",
        "Source": "https://github.com/yourusername/chain-of-debate",
        "Documentation": "https://github.com/yourusername/chain-of-debate/wiki",
    },
)
