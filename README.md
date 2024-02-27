# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for the marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pooja.S
RegisterNumber: 212223040146
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
X=df.iloc[:,0:1]
Y=df.iloc[:,-1]
X
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:
![Screenshot 2024-02-27 072542](https://github.com/poojasen05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150784373/9a74b3e5-e1e7-47c6-992f-93292bb0d7a0)
![Screenshot 2024-02-27 072653](https://github.com/poojasen05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150784373/7ed2ef6b-bb56-4bc1-abe1-8c37bc791455)
![Screenshot 2024-02-27 072719](https://github.com/poojasen05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150784373/28bf2e82-f25c-403d-b17b-6dc44cb313c7)
![Screenshot 2024-02-27 072749](https://github.com/poojasen05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150784373/31acd580-6ad8-4207-bfe3-e20e1dd1879f)
![Screenshot 2024-02-27 072925](https://github.com/poojasen05/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150784373/9ff58bb0-eab3-44fb-8415-155bb6f9479f)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
