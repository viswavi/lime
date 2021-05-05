from setuptools import setup, find_packages

setup(name='contrastlime',
      version='0.2.0.1',
      description='Contrastive Local Interpretable Model-Agnostic Explanations for comparing machine learning classifiers',
      url='http://github.com/viswavi/lime',
      license='BSD',
      packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.5',
      install_requires=[
          'matplotlib',
          'numpy',
          'scipy',
          'tqdm >= 4.29.1',
          'scikit-learn>=0.18',
          'scikit-image>=0.12',
          'pyDOE2==1.3.0'
      ],
      extras_require={
          'dev': ['pytest', 'flake8'],
      },
      include_package_data=True,
      zip_safe=False)
