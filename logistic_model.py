import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import math
import typing

# Importing Data
df = pd.read_csv("data/debug.csv")

# Preprocessing
df.dropna(inplace=True)

# separating targets and features
features = ['Gender', 'Married', 'Dependents', 'Education', 'Income',
            'Coapplicant_Income', 'Loan_Tenor', 'Credit_History', 'Property_Area']
targets = ['Loan_Status']

x = df[features]
y = df[targets]

# shuffling and splitting data in testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# # categorical features are encoded using one-hot encoding
# categorical_columns = x.select_dtypes(include=['object']).columns
# X_train_encoded = pd.get_dummies(
#     X_train, columns=categorical_columns, drop_first=True)
# X_test_encoded = pd.get_dummies(
#     X_test, columns=categorical_columns, drop_first=True)

# # encode categorical targets
# le = LabelEncoder()
# y_train_encoded = le.fit_transform(y_train)
# y_test_encoded = le.transform(y_test)

# # numerical features are standardized
# numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
# scaler = StandardScaler()
# X_train_encoded[numerical_columns] = scaler.fit_transform(
#     X_train_encoded[numerical_columns])
# X_test_encoded[numerical_columns] = scaler.transform(
#     X_test_encoded[numerical_columns])

# z = w.x + b
# fw,b(x) = g(z) = 1/(1+e**(-z))
# returns one value not a list


def get_z(w, x, b):
    return np.dot(w, x) + b

# sigmoid: g(z)
# returns one value not a list


def get_sigmoid(z):
    return 1/(1+np.exp(-z))


def get_sigmoid_values(w, x, b):
    m = x.shape[0]
    sigmoid_values = np.zeros(m)
    for i in range(m):
        z = get_z(w, x[i], b)
        sigmoid_values[i] = get_sigmoid(z)
    return sigmoid_values

# Logistic Regression


def logistic_regression(x, y):
    # hypothesis function

    x = x.to_numpy()
    y = y.to_numpy().flatten()

    rows, cols = x.shape[0], x.shape[1]
    w = np.zeros((1, cols))
    b = 0

    sigmoid_values = get_sigmoid_values(w, x, b)

    # cost function:
    # L(fw,b(xi), y) = -ylog(fw,b(x)) - (1-y)log(1-fw,b(x))
    # J = -1/m * sigma (-ylog(fw,b(x)) - (1-y)log(1-fw,b(x)))
    cost = (-1/rows) * np.sum(-y * np.log10(sigmoid_values) -
                              (1-y)*np.log10(1-sigmoid_values))

    # gradient descent
    alpha = 0.001
    max_iterations = 100
    for i in range(max_iterations):
        h = get_sigmoid_values(w, x, b)
        print(h, x[i], y)
        new_w = np.zeros((cols))
        new_b = 0
        # for j in range(cols):
        #     dw = (1/rows) * np.sum((h-y)*x[j])
        #     db = (1/rows) * np.sum(h-y)
        #     new_w = w-alpha*dw
        #     new_b = b-alpha*db
        # w = new_w
        # b = new_b
    print(w, b)

    # how to predict? should we use f(x) or sigmoid?


# Accuracy


# Training


# Predict


# debugging
logistic_regression(x, y)
