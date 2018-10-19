from setuptools import setup, find_packages
import versioneer

setup(
    name='deepmachine',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Deep Learning Framework',
    packages=find_packages(),
    author='Yuxiang ZHOU',
    author_email='mr.yuxiang.zhou@googlemail.com',
    url='https://github.com/yuxiang-zhou/deepmachine',
    install_requires=[
        'cython',
        'numpy',
        'scikit-image',
        'scikit-learn',
        'tensorflow-gpu',
        'Keras>=2.2.2',
    ]
)
