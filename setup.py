import os
import setuptools


def requirements():
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
        return f.read().splitlines()


setuptools.setup(
    name="ratsnlp",
    version="2023.03.21",
    license='MIT',
    author="ratsgo, chrisjihee",
    author_email="ratsgo@naver.com, chrisjihee@naver.com",
    description="tools for Natural Language Processing",
    long_description=open('README.md', encoding='utf-8').read(),
    url="https://github.com/chrisjihee/ratsnlp",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'ratsnlp.nlpbook.classification': ['*.html'],
        'ratsnlp.nlpbook.ner': ['*.html'],
        'ratsnlp.nlpbook.qa': ['*.html'],
        'ratsnlp.nlpbook.paircls': ['*.html'],
        'ratsnlp.nlpbook.generation': ['*.html'],
    },
    install_requires=requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)