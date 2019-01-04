# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
Script for training the BIMPM.
"""

import tensorflow as tf
import logging
import numpy as np
import argparse
import bimpm
import os 
import shutil
from config import Config
def initialize(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

if __name__ == "__main__":
    

    logging.basicConfig(level=logging.INFO)

    sess = tf.Session()
    
    logging.info('Reading training data')
    train_p = np.random.choice(np.arange(1,20), size = (100,5))
    train_h = np.random.choice(np.arange(1,20), size = (100,5))
    train_y = np.random.randint(2, size = 100)
    train_data = (train_p, train_h, train_y)

    logging.info('Reading validation data')
    valid_p = np.random.choice(np.arange(1,20), size = (80,5))
    valid_h = np.random.choice(np.arange(1,20), size = (80,5))
    valid_y = np.random.randint(2, size = 80)
    valid_data = (valid_p, valid_h, valid_y)
    
    logging.info('Creating model')

    initialize(Config["save_dir"])
    model = bimpm.BIMPM(Config)
    
    sess.run(tf.global_variables_initializer())
    model.show_parameter_count(model.get_trainable_variables())
    logging.info('Initialized the model and all variables. Starting training.')
    model.train(sess, Config["save_dir"], train_data, valid_data, Config["batch_size"],
                Config["epochs"])
    
    
    
