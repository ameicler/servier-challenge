from setuptools import setup, find_packages

setup(
    author="Antoine Meicler",
    description="Servier ML Technical Test",
    name='servier-pkg-ameicler',
    author_email="a.meicler@gmail.com",
    python_requires='>=3.6',
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        'console_scripts': [
            "servier=utils.cli:cli"
        ],
    }
)
