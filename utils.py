import numpy as np 
import os 
import tensorflow as tf 
import random
from sklearn.utils import shuffle

def data_generator(batch_size, data):
    p, h, y = data

    p = np.array(p)
    h = np.array(h)
    y = np.array(y)
    size = y.shape[0]
    batches = (size + batch_size - 1) // batch_size

    for i in range(batches):
        p_batch = p[i* batch_size: min((i+1) * batch_size, size)]
        h_batch = h[i* batch_size: min((i+1) * batch_size, size)]
        y_batch = y[i* batch_size: min((i+1) * batch_size, size)]
        yield p_batch, h_batch, y_batch

def shuffle_data(data):
    p, h, y = data

    p = np.array(p)
    h = np.array(h)
    y = np.array(y)

    idx = np.arange(y.shape[0])
    random.shuffle(idx)
    
    p = p[idx]
    h = h[idx]
    y = y[idx]

    return (p, h, y)

