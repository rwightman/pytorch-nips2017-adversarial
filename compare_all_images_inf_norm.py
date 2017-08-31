# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import os
import time

look_dir = '/tmp/nips/attacks/'
src_dir = '/home/aleksey/code/nips2017-nw/dataset/images/'
look_image = '0f9d8c86a9f38020.png'

for d in sorted(os.listdir(look_dir)):
    with open(os.path.join(look_dir,d,look_image), 'rb') as f:
        image1 = Image.open(f).convert('RGB')
        image1 = np.array(image1).astype(np.float)
    
    with open(os.path.join(src_dir,look_image), 'rb') as f:
        image2 = Image.open(f).convert('RGB')
        image2 = np.array(image2).astype(np.float)
    
    dif = image2-image1
    abs_dif = np.abs(image2-image1)

    try:
        print('{} - {} - {}'.format(
            np.max(abs_dif),
            d,
            time.strftime('%m/%d/%Y', time.gmtime(os.path.getmtime(os.path.join(look_dir,d,'0be391239ccba0f2.png'))))))
    except:
        print('{} - {}'.format(
            np.max(abs_dif),
            d))


"""
np.max(abs_dif)
np.min(abs_dif)
np.mean(abs_dif)

np.max(dif)
np.min(dif)
np.mean(dif)
"""