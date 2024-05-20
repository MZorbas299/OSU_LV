import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Učitavanje podataka
data = pd.read_csv("winequality-red.csv")

# Broj vina
broj_vina = len(data)
print("Broj vina provedenih mjerenje:", broj_vina)
# Priprema histograma
plt.hist(data['alcohol'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribucija alkoholne jakosti vina')
plt.xlabel('Alkoholna jakost')
plt.ylabel('Broj uzoraka')
plt.show()
# Broj uzoraka vina s kvalitetom manjom od 6
manje_od_6 = data[data['quality'] < 6]['quality'].count()

# Broj uzoraka vina s kvalitetom 6 ili većom
6_ili_veca = data[data['quality'] >= 6]['quality'].count()

print("Broj uzoraka vina s kvalitetom manjom od 6:", manje_od_6)
print("Broj uzoraka vina s kvalitetom 6 ili većom:", 6_ili_veca)
# Izračun korelacije
korelacija = data.corr()

# Prikaz korelacije pomoću heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(korelacija, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelacija svih veličina u datasetu')
plt.show()

////////////////////////////////////////////////////////////////////////////

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

# Učitavanje podataka
data = pd.read_csv("winequality-red.csv")

# Zamjena vrijednosti kvalitete vina
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Podijela na ulazne i izlazne podatke
X = data.drop('quality', axis=1)
y = data['quality']

# Podjela na skup za učenje i skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizacija podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Izgradnja linearnog regresijskog modela
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Ispisivanje parametara modela
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
# Procjena izlazne veličine
y_pred = model.predict(X_test_scaled)

# Prikaz dijagrama raspršenja
plt.scatter(y_test, y_pred)
plt.xlabel('Stvarna kvaliteta')
plt.ylabel('Predviđena kvaliteta')
plt.title('Procjena izlazne veličine vs Stvarna izlazna veličina')
plt.show()

# Računanje regresijskih metrika
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Ispisivanje rezultata
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)
print("R2 score:", r2)