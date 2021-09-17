# --- built in ---
import os
import re

from setuptools import find_packages, setup

def get_version():
    with open(os.path.join('unstable_baselines', '__init__.py'), 'r') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

setup(
    name='unstable_baselines',
    version=get_version(),
    description='A TensorFlow 2.0 implemented RL baselines',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Ending2015a/unstable_baselines',
    author='JoeHsiao',
    author_email='joehsiao@gapp.nthu.edu.tw',
    license='MIT',
    python_requires=">=3.6",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='reinforcement-learning playform tensorflow-2.0 tensorflow',
    packages=[
        # exclude deprecated module
        package for package in find_packages(exclude=["*.dep.*", "*.dep"])
        if package.startswith('unstable_baselines')
    ],
    package_data={
        'unstable_baselines.logger': ['ascii_font/pkl/*.pkl']
    },
    install_requires=[
        'gym==0.18.0',
        'tensorflow>=2.3.0',
        'numpy',
        'cloudpickle',
        'opencv-python',
        'tqdm'
    ],
    extras_require={
        'dev': [
            # Unittest and coverage
            'coverage',
            'parameterized',
            'scipy',
            'yapf',
            'flake8',
            'flake8-bugbear',
        ],
        'atari': ['atari_py==0.2.6'],
        'pybullet': ['pybullet'],
        'extra': [
            'atari_py==0.2.6',
            'pybullet',
            'ray[tune]'
        ]
    }
)
