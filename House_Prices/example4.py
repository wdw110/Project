#encoding=utf-8

#http://blog.nycdatascience.com/student-works/kaggle-competition-2017-house-price-prediction/

import numpy as np
import pandas as pd
import xgboost as xgb

# Load the data.
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df["MasVnrArea"].fillna(0, inplace=True)


