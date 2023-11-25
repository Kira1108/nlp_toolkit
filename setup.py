from setuptools import setup
from setuptools import find_packages

requires = ["transformers","datasets","accelerate","evaluate",'googletrans',
 "tiktoken","langchain","chromadb","torch",
 "tensorflow","sentencepiece","nltk","openai==1.3.3","llama-index"]

setup(name='nlp_toolkit',
      version='0.0.1',
      description='use nlp toolkit to power your nlp projects',
      author='The fastest man alive.',
      packages=find_packages(),
      install_requires=requires)






