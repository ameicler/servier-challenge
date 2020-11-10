from setuptools import setup

setup(
    author="Antoine Meicler",
    description="Servier ML Technical Test",
    name='servier',
    entry_points={
        'console_scripts': [
            "servier=src.utils.cli:cli"
        ],
    }
)
