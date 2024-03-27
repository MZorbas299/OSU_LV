import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("data_C02_emission.csv")
#print(data)
input_variable = ["Engine Size (L)", "Cylinders", "Fuel Consumption City (L/100km)", "Fuel Consumption Hwy (L/100km)", "Fuel Consumption Comb (L/100km)", "Fuel Consumption Comb (mpg)", "Fuel type"]

X = data[input_variable]
y = data["CO2 Emissions (g/km)"]
ohe = OneHotEncoder ()
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()
X["Fuel Type"] = X_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

plt.xlabel("Input variable")
plt.ylabel("CO2 Emissions")
plt.scatter(x=X_train["Fuel Consumption City (L/100km)"], y = y_train, c="Blue")
plt.scatter(x=X_test["Fuel Consumption City (L/100km)"], y = y_test, c="Red")
plt.show()

sc = MinMaxScaler ()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)
plt.xlabel("Input variable")
plt.hist(x=X_train["Fuel Consumption City (L/100km)"], bins = 5)
#plt.show()
plt.hist(x=X_train_n[:, 1], bins = 5)
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n , y_train)
print(linearModel.coef_)

y_test_p = linearModel.predict(X_test_n)
plt.xlabel("Real values")
plt.ylabel("Predicted values")
plt.scatter(x=y_test, y=y_test_p)
plt.show()