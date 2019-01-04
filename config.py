import numpy as np 
import pandas as pd 
import os
import sys

Config = {
    "word_vocab_size": 100,
    "word_dim": 100,
    "train": True,
    
    "use_char_emb": False,
    "max_word_len": 10,
    "char_vocab_size": 50,
    "char_dim": 64,
    "char_hidden_size": 64,

    "hidden_size": 64,

    "class_size": 2,

    "dropout": 0.2,
    "learning_rate": 0.001,
    "clip_value": 5.0,

    "epochs": 10,
    "batch_size": 64,

    "save_dir": "./output",



    "num_perspective": 10




}