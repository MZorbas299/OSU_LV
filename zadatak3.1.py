# Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv.
#Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:
#a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili
#duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke velicine konvertirajte u tip
# category.
#b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal:
#ime proizvodaca, model vozila i kolika je gradska potrošnja.
#c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija
#plinova za ova vozila?
#d) Koliko mjerenja se odnosi na vozila proizvo¯daca Audi? Kolika je prosjecna emisija C02
#plinova automobila proizvodaca Audi koji imaju 4 cilindara?
#e) Koliko je vozila s 4,6,8... cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na
#broj cilindara?
#f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila
#koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
#g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?
#h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)?
#i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat


import pandas as pd

data=pd.read_csv('data_C02_emission.csv')


#a)
print(len(data)) 
print(data.describe())
print(data.isnull().sum())
data.drop_duplicates()
data=data.reset_index(drop=True)

for col in ['Make', 'Model', 'Vehicle Class', 'Transmission','Fuel Type']:
    data[col] = data[col].astype('category')

print(data.info())

#b)
print("Prva tri koja najvise trose:")
print(data.sort_values(by="Fuel Consumption City (L/100km)").head(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print("Prva tri koja najmanje trose:")
print(data.sort_values(by="Fuel Consumption City (L/100km)").tail(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#c)

velicinamotora = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print("Broj vozila koji imaju velicinu motora izmedu 2.5 i 3.5L:",len(velicinamotora))
print("Prosjecna emisija vozila:",velicinamotora['CO2 Emissions (g/km)'].mean())

#d)
audi=data[(data['Make']=="Audi")]
print("Broj mjerenja Audi vozila iznosi:",len(audi))
print(f'Prosječna emsija audija s 4 cilindra',audi[(audi['Cylinders'] == 4)][['CO2 Emissions (g/km)']].mean())

#e)
cc = data["Cylinders"].value_counts()

print(f"Broj vozila po cilindrima iznosi: ",cc)

avgco2 = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
print(avgco2)



#f)
dizel = data[(data['Fuel Type'] == 'D')]
print(dizel['Fuel Consumption City (L/100km)'].mean(),'je prosjecna gradska potrošnja za vozila koja koriste dizel')
print(dizel.median())
benzin = data[(data['Fuel Type'] == 'Z')]
print(benzin['Fuel Consumption City (L/100km)'].mean(),'je prosjecna gradska potrošnja za vozila koje koriste benzin')
print(benzin.median())


#g) 
cetiricil = data[(data["Cylinders"]==4) & (data["Fuel Type"]=='D')]
print(f"Vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva",cetiricil.sort_values(ascending=False, by='Fuel Consumption City (L/100km)')[['Make', 'Model']].head(1))

#h)
rucni = data[data["Transmission"].str.contains("M")]
print("Broj vozila s rucnim mjenjacem ",len(rucni))

#i)  
print (f"Korelacija izmedu numerickih velicina \n", data.corr(numeric_only = True))




