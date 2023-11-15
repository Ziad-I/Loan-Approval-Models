import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
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


# Logistic Regression
def logistic_regression(x, y):
    # hypothesis function

    x = x.to_numpy()

    rows, cols = x.shape[0], x.shape[1]
    w = np.zeros((cols, 1))
    b = 0

    sigmoid = np.zeros((rows))
    for i in range(rows):
        # calculate z
        z = np.dot(w.T, x[i]) + b
        # sigmoid
        sigmoid[i] = 1/(1+np.exp(-z))

    # cost function

    # gradient descent

    def gradient_descent():
        # calculate dw, db
        pass

    # how to predict? should we use f(x) or sigmoid?


# Accuracy


# Training


# Predict


# debugging
logistic_regression(x, y)
