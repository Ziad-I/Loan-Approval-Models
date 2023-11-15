import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('./data/loan_old.csv')

print(data.isna().sum())

print(data.dtypes)

for col in data:
  if data[col].dtype != 'object':
    sns.boxplot(data = data,x=col)
    plt.show()


sns.pairplot(data, diag_kind='kde')  # 'kde' for kernel density estimation
plt.show()