#!/usr/bin/env python

from distutils.core import setup

setup(name='EEGAnalysis',
      version='0.8.1',
      description='EEG Analysis for Zhang\'s Lab',
      author='Yizhan Miao',
      author_email='yzmiao@pm.me',
      packages=['EEGAnalysis', 'EEGAnalysis.decomposition',
                'EEGAnalysis.io'],
     )
