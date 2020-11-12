from setuptools import setup

setup(
    author="Antoine Meicler",
    description="Servier ML Technical Test",
    name='servier-pkg-ameicler',
    author_email="a.meicler@gmail.com",
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            "servier=src.utils.cli:cli"
        ],
    }
)
