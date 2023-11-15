import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import typing


def sanitize_data(path):
    data = pd.read_csv('./data/loan_old.csv')
    # data.info()

    # dropping rows with missing values
    data.dropna(inplace=True)

    # replacing coapplicant_income with values 0 with the mean value
    data['Coapplicant_Income'] = data['Coapplicant_Income'].replace(0, data[data['Coapplicant_Income'] == 0][
        'Coapplicant_Income'].mean())

    # removing outliers in the income
    data.drop(data[data['Income'] > 25000].index, axis=0, inplace=True)

    return data


clean_data = sanitize_data("'./data/loan_old.csv")

# separating targets and features
features = ['Gender', 'Married', 'Dependents', 'Education', 'Income', 'Coapplicant_Income', 'Loan_Tenor',
            'Credit_History', 'Property_Area']
targets = ['Loan_Status']

x = clean_data[features]
y = clean_data[targets]

# shuffling and splitting data in testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

label_encoder = LabelEncoder()

# categorical data are encoded
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = label_encoder.fit_transform(X_train[col])

for col in y_train.columns:
    if y_train[col].dtype == 'object':
        y_train[col] = label_encoder.fit_transform(y_train[col])

for col in X_test.columns:
    if X_test[col].dtype == 'object':
        X_test[col] = label_encoder.fit_transform(X_test[col])

for col in y_test.columns:
    if y_test[col].dtype == 'object':
        y_test[col] = label_encoder.fit_transform(y_test[col])

# numerical features are standardized
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# print(X_train,y_train)


def get_z(x, w, b):
    return np.dot(x, w) + b

# sigmoid: g(z)


def get_sigmoid(z):
    return 1/(1+np.exp(-z))


# cost function:
# L(fw,b(xi), y) = -ylog(fw,b(x)) - (1-y)log(1-fw,b(x))
# J = -1/m * sigma (-ylog(fw,b(x)) - (1-y)log(1-fw,b(x)))
def get_cost(y, predictions):
    m = len(y)
    cost = (-1/m) * np.sum(-y * np.log(predictions) -
                           (1-y)*np.log(1-predictions))
    return cost

# Logistic Regression


def logistic_regression(x, y):

    x = x.to_numpy()
    y = y.to_numpy()

    m, features = x.shape[0], x.shape[1]
    # fx1
    w = np.zeros((features, 1))
    b = 0

    # gradient descent
    alpha = 0.001
    max_iterations = 100

    for i in range(max_iterations):
        # mxf . fx1 = mx1
        z = get_z(x, w, b)
        predictions = get_sigmoid(z)

        # fxm . mx1 = fx1
        dw = (1/m) * np.dot(x.T, predictions - y)
        db = (1/m) * np.sum(predictions - y)

        new_w = w - alpha * dw
        new_b = b - alpha * db
        w = new_w
        b = new_b

    return w, b

    # how to predict? should we use f(x) or sigmoid?


# Accuracy


# Training


# Predict


# debugging
print(logistic_regression(x, y))
