# Regression-in-Pandas
#by using a dataset how we can do regression via python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Load the dataset
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Basic data overview
print(iris.describe())
print(iris.info())

# (Optional) Convert 'petal_length' into bins if you intend to use get_dummies
iris['petal_length_category'] = pd.cut(iris['petal_length'], bins=3, labels=False)
dummies = pd.get_dummies(iris, columns=['petal_length_category'], dtype="int64")
print(dummies)

# Dropping the 'petal_width' column as an example
datal = dummies.drop(columns=["petal_width"])
print(datal)

# Display column correlations
print(datal.corr().to_string())

# Define X (features) and Y (target)
# Assuming you want to predict 'sepal_length' as an example
X = datal.drop(columns=["sepal_length", "species"])
Y = datal['sepal_length']
print(X)
print(Y)

# Scatterplot
sns.scatterplot(x='sepal_width', y='sepal_length', data=iris)
plt.show()

# Train-test split
random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Polynomial features transformation (optional)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Linear regression
regr = LinearRegression()
regr.fit(X_train_poly, Y_train)

# Model evaluation
print("Polynomial Regression R^2 Score:", regr.score(X_test_poly, Y_test))
