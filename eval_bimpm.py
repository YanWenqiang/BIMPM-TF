# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
Script for evaluate the BIMPM.
"""

import tensorflow as tf
import logging
import numpy as np
import bimpm
import os 
import shutil
from config import Config

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)

    sess = tf.Session()
    

    logging.info('Reading validation data')
    valid_p = np.random.choice(np.arange(1,20), size = (80,5))
    valid_h = np.random.choice(np.arange(1,20), size = (80,5))
    valid_y = np.random.randint(2, size = 80)
    valid_data = (valid_p, valid_h, valid_y)
    
    logging.info('Creating model')

    
    model = bimpm.BIMPM.load(Config["save_dir"], sess)
    
    model.predict(sess, Config["save_dir"], valid_data)