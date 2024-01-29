import os
import numpy as np 
import pickle

with open('results/hrsc/r3det/r50_orig_train_finetune_3x/one_img.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)