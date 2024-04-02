from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="controlgym",
    version="1.0.0",
    description="large-scale safety-critical control benchmarks for reinforcement learning algorithms",
    license="MIT",
    long_description=long_description,
    author="Xiangyuan Zhang, Weichao Mao, Saviz Mowlavi, Mouhacine Benosman, and Tamer Basar",
    author_email="xz7@illinois.edu",
    packages=["controlgym"],
    package_data={
        "controlgym.envs": ["linear_control_src/*.mat"],
    },
    install_requires=[
        "numpy>=1.23.5",
        "scipy>=1.11.1",
        "matplotlib>=3.7.1",
        "torch>=2.0.1",
        "gymnasium==0.29.1",
    ],
)
