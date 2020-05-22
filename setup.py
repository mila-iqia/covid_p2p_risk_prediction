from setuptools import setup

setup(
    name='ctt',
    version='0.1',
    packages=['ctt', 'ctt.models', 'ctt.inference', 'ctt.conversion', 'ctt.data_loading'],
    url='https://github.com/nasimrahaman/ctt',
    license='MIT',
    author='Nasim Rahaman',
    author_email='nasim.rahaman@tuebingen.mpg.de',
    description='Contact Tracing Transformer'
)
