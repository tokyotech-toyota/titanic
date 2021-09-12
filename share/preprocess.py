#!/usr/bin/env python3

################################################################################
################################################################################
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer
from collections import namedtuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# import tensorflow as tf

################################################################################
################################################################################
################################################################################

# [INFO]
#   Insert line below where you want to debug interactively:
#   import ipdb; ipdb.set_trace()
#
# [URL]
#   https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic/notebook

# [DATASET]
#   https://qiita.com/suzumi/items/8ce18bc90c942663d1e6
#
#   * Pclass
#   * Sex
#   * Age

################################################################################
################################################################################
################################################################################

# feature engineering
def nan_padding(data, columns):
    for column in columns:
        imputer=SimpleImputer()
        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))
    return data

################################################################################

def drop_not_concerned(data, columns):
    return data.drop(columns, axis=1)

################################################################################

# load data
train_data = pd.read_csv(r"dat/train.csv")
test_data = pd.read_csv(r"dat/test.csv")

nan_columns = ["Age", "SibSp", "Parch"]

train_data = nan_padding(train_data, nan_columns)
test_data = nan_padding(test_data, nan_columns)

#save PassengerId for evaluation
test_passenger_id=test_data["PassengerId"]

#not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]
not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked", "SibSp", "Parch"]

train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)

def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data

dummy_columns = ["Pclass"]

train_data=dummy_data(train_data, dummy_columns)
test_data=dummy_data(test_data, dummy_columns)

def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male","female"])
    data["Sex"]=le.transform(data["Sex"]) 
    return data

train_data = sex_to_int(train_data)
test_data = sex_to_int(test_data)

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data
train_data = normalize_age(train_data)
test_data = normalize_age(test_data)


def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y

train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))
print("train_y content:{}".format(train_y[:3]))

print("valid_x:{}".format(valid_x.shape))
print("valid_y:{}".format(valid_y.shape))

print(train_x)  

clf = LogisticRegression()

clf.fit(train_x, train_y)

dummy_ans = clf.score(train_x, train_y)
print(dummy_ans)

import ipdb; ipdb.set_trace()
