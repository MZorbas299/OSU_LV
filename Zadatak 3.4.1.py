import pandas as pd
import numpy as py

#a)
data = pd.read_csv("data_C02_emission.csv")

print(data.info())
print(len(data))
print(data.isnull().sum())

data["Make"] = data["Make"].astype("category")
data["Model"] = data["Model"].astype("category")
data["Vehicle Clas"] = data["Vehicle Class"].astype("category")
data["Transmission"] = data["Transmission"].astype("category")
data["Fuel Type"] = data["Fuel Type"].astype("category")

#b)
sorted_data = data.sort_values(by=["Fuel Consumption City (L/100km)"])
print("Najveca potrosnja: ")
print(sorted_data[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3))
print("Najmanja potrosnja: ")
print(sorted_data[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3))

#c)
new_data = data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)]
print(len(new_data))
print(new_data["CO2 Emissions (g/km)"].mean())

#d)
data_audi = data[data["Make"] == "Audi"]
print(len(data_audi))
data_audi4 = data[data["Cylinders"] == 4]
print(data_audi4["CO2 Emissions (g/km)"].mean())

#e)
data4 = data[data["Cylinders"] == 4]
print(len(data4))
print(data4["CO2 Emissions (g/km)"].mean())
data6 = data[data["Cylinders"] == 6]
print(len(data6))
print(data6["CO2 Emissions (g/km)"].mean())
data8 = data[data["Cylinders"] == 8]
print(len(data8))
print(data8["CO2 Emissions (g/km)"].mean())

#f)
data_dizel = data[data["Fuel Type"] == "D"]
data_benzin = data[data["Fuel Type"] == "X"]
print(data_dizel["Fuel Consumption City (L/100km)"].mean())
print(data_benzin["Fuel Consumption City (L/100km)"].mean())
print(data_dizel["Fuel Consumption City (L/100km)"].median())
print(data_benzin["Fuel Consumption City (L/100km)"].median())

#g)
data_diz4 = (data[data["Cylinders"] == 4]) & (data[data["Fuel Type"] == "D"])
print() 
