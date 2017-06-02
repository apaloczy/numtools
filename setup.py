from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='numtools',
      version='0.1',
      description='Simple implementations of some numerical methods, mostly solvers for differential equations.',
      url='https://github.com/apaloczy/numtools',
      license='MIT',
      packages=['numtools'],
      install_requires=['numpy'],
      test_suite = 'nose.collector',
      zip_safe=False)
