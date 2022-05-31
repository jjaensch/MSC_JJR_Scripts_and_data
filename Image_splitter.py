# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:02:18 2022
@author: Jakob Jaensch Rasmussen
"""

import os
from image_slicer import slice

images = []
directory = os.getcwd()
for file in os.listdir():
    f = os.path.join(directory, file)
    # checking if it is a file
    images.append(f)

slice('DJI_0002.JPG',col=8,row=6)