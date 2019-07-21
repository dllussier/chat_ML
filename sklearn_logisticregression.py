'''
logistic regrassion for age group prediction based on language usage in chat rooms
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#designate input file
input_file = "chatdata.csv"

#pandas read input csv
df = pd.read_csv(input_file, header = 0)

#select data
X = data[:, 1:]  #select columns 2 through end, predictors
y = data[:, 1]   #select column 0, target

