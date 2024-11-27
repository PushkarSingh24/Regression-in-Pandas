# Regression-in-Pandas
#by using a dataset how we can do regression via python

import pandas as pd 
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv') 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split 
import random as random  
print(iris.describe()) 
print(iris.info()) 
dummies=pd.get_dummies(iris,columns=['petal_length'], dtype="int64") 
print(dummies) 
datal = dummies.drop(columns=["petal_width"]) 
print(data1) 
print(datal.columns) 
print(datal.corr().to_string()) 
X=datal[['sepal_length' ,'sepal_width' ,'petal_length' ,'petal_width']] 
Y=datal['Count'] 
print(X) 
print(Y) 
sns.scatterplot(iris=datal) 
plt.plot(X,Y) 
random.seed(1) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y) 
regr = LinearRegression() 
regr.fit(X_train, Y_train) 
regr.score(poly.transform(X_test,Y_test)) 
print(regr.score(X_test,Y_test))
