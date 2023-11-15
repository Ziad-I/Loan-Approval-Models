import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def sanitize_data(path):
    data = pd.read_csv('./data/loan_old.csv')
    # data.info()

    # dropping rows with missing values
    data.dropna(inplace=True)

    # replacing coapplicant_income with values 0 with the mean value
    data['Coapplicant_Income'] = data['Coapplicant_Income'].replace(0, data[data['Coapplicant_Income'] == 0]['Coapplicant_Income'].mean())
    
    # removing outliers in the income
    data.drop(data[data['Income']>25000].index,axis=0,inplace=True)
    
    return data


clean_data = sanitize_data("'./data/loan_old.csv")

# separating targets and features
features = ['Gender', 'Married', 'Dependents', 'Education', 'Income', 'Coapplicant_Income', 'Loan_Tenor', 'Credit_History', 'Property_Area']
targets = ['Max_Loan_Amount']

x = clean_data[features]
y = clean_data[targets]

# shuffling and splitting data in testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# categorical features are encoded using one-hot encoding
categorical_columns = x.select_dtypes(include=['object']).columns
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# numerical features are standardized
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X_train_encoded[numerical_columns] = scaler.fit_transform(X_train_encoded[numerical_columns])
X_test_encoded[numerical_columns] = scaler.transform(X_test_encoded[numerical_columns])


def linear_train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def linear_predict(model, x):
    predictions = model.predict(x)
    return predictions
    
def r2_evaluate(yhat, y):
    r2 = r2_score(yhat, y)
    print('r2 score for this linear regression model is', r2)
 
def linear_regression():
    # load old data
    # train model on train part of old data
    model = linear_train(X_train_encoded, y_train)
    # test model on test part of old data
    predictions_test = linear_predict(model, X_test_encoded)
    # evaluate the model
    r2_evaluate(predictions_test, y_test)
    
    # #load new data and preprocess it (encoding and standardization)
    new_data = sanitize_data("data/loan_new.csv")
    X_new = new_data[features]
    X_new = pd.get_dummies(X_new, columns=categorical_columns, drop_first=True)
    X_new[numerical_columns] = scaler.transform(X_new[numerical_columns])
    # predict for new data
    predictions_new = linear_predict(model, X_new)
    print(predictions_new)

if __name__ == "__main__":
    linear_regression()
