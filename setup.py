from setuptools import setup, find_packages

setup(
    name='SMI_analysis',
    version='1.0',
    py_modules=['SMI_beamline', 'remesh', 'stitch', 'integrate1D', 'Detector'],
    install_requires=['numpy', 'scipy', 'pyFAI'],
    url='',
    license='',
    author='gfreychet',
    author_email='gfreychet@gmail.com',
    description='SMI analysis tools'
)