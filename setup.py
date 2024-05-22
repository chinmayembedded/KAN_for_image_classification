from setuptools import setup, find_packages

setup(
    name='kan_classifiers',
    version='0.1',
    packages=find_packages(),
    description='KAN Classifiers: Benchmarking and Comparison with SOTA Architectures',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chinmay Sawant, Tommaso Capecchi',
    author_email='chinmayssawant@live.com, tommycaps@hotmail.it',
    url='https://github.com/chinmayembedded/KAN_for_image_classification.git',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'matplotlib',
        'torchinfo',
        'timm',
        'datasets'
    ],
    python_requires='>=3.10',
)
