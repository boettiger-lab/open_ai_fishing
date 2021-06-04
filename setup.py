import setuptools

setuptools.setup(
    name="gym_fishing",
    version="0.1.0",
    license="MIT",
    description="Provide gym environments for reinforcement learning",
    author="Carl Boettiger & Marcus Lapeyrolerie",
    author_email="cboettig@gmail.com",
    url="https://github.com/boettiger-lab/gym_fishing",
    keywords=[
        "RL",
        "Reinforcement Learning",
        "Conservation",
        "stable-baselines",
        "OpenAI Gym",
        "AI",
        "Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=["docs", "scripts", "tests"]),
    install_requires=["gym", "numpy", "pandas", "matplotlib"],
    extras_require={
        "tests": [
            "stable-baselines3",
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
        "extra": ["twine"],
    },
    # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
