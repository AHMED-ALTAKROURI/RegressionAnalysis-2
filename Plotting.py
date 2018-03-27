from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as mn


dataset = read_csv('C://Users/Ahmed/Desktop/backup/MyProject/ML-EXP/Regresstion/dataset_1.csv', header=None)


# lets first normalize and replace missing values with mean of the data
dataset.fillna(0, inplace=True)
print(dataset.isnull().sum())


# change feature you want to plot and correlate to the target value
features = dataset[[1]]
targets = dataset[[23]]

print(features.values)
print(targets.values)


plt.plot(features.values.tolist())
plt.ylabel(targets.values.tolist())
plt.show()

