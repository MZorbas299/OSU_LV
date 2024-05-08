import numpy as np
import matplotlib . pyplot as plt
import pandas as pd

#a) Na temelju velicine numpy polja data,
# na koliko osoba su izvršena mjerenja?
data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
osoba = data.shape[0]
print(osoba)

#b) Postoje li izostale ili duplicirane vrijednosti
# u stupcima s mjerenjima dobi i indeksa tjelesne
#mase (BMI)? Obrišite ih ako postoje. Koliko je sada
#uzoraka mjerenja preostalo?
izostale = np.isnan(data[:, 2]).any()
if izostale:
    print("postoje izostale")
else:
    print("ne postoje izostale")
dupl = len(np.unique(data[:, 0])) != len(data[:, 0])
if dupl:
    print("postoje dupl")
else:
    print(" ne postoje dupl")
data_bez_izos = data[~np.isnan(data[:, 2])]
data_bez_dupl = np.unique(data_bez_izos, axis=0)
broj = data_bez_dupl.shape[0]
print(broj)

#c) Prikažite odnos dobi i indeksa tjelesne mase (BMI)
# osobe pomocu scatter dijagrama.
#Dodajte naziv dijagrama i nazive osi s pripadajucim mjernim jedinicama
plt.scatter(data[:, 1], data[:, 2])
plt.title('Odnos visine i mase osobe')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.show()

#d) Izracunajte i ispišite u terminal minimalnu, maksimalnu
# i srednju vrijednost indeksa tjelesne
#mase (BMI) u ovom podatkovnom skupu.
print("Min: " + str(data[:, 1].min()))
print("Max: " + str(data[:, 1].max()))
print("Srednja: " + str(data[:, 1].mean()))

#e) Ponovite zadatak pod d), ali posebno za osobe kojima
# je dijagnosticiran dijabetes i za one
#kojima nije. Kolikom je broju ljudi dijagonosticiran dijabetes?
indm = (data[:, 0] == 1)
print("Min: " + str(data[indm, 1].min()))
print("Max: " + str(data[indm, 1].max()))
print("Srednja: " + str(data[indm, 1].mean()))

indn = (data[:, 0] == 0)
print("Min: " + str(data[indn, 1].min()))
print("Max: " + str(data[indn, 1].max()))
print("Srednja: " + str(data[indn, 1].mean()))

osobe_s_dij = np.sum(data[:, -1] == 71.93) #-1 znaci ide u zadnji
#stupac odma
print(osobe_s_dij)

#22222222222222222222222222222222222222222222222222222222222222222222222222222222
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

# Učitavanje podataka
data = np.genfromtxt('data.csv', delimiter=',', skip_header=True)

# Podjela na ulazne podatke X i izlazne podatke y
X = data[:, :-1]  # Ulazni podaci su svi stupci osim zadnjeg
y = data[:, -1]   # Izlazni podaci su zadnji stupac

# Podjela podataka na skup za učenje i skup za testiranje (omjer 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# a) Izgradnja modela logističke regresije
model = LogisticRegression()
model.fit(X_train, y_train)

# b) Provođenje klasifikacije skupa podataka za testiranje
y_pred = model.predict(X_test)

# c) Izračunavanje i prikazivanje matrice zabune na testnim podacima
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
conf_matrix_display.plot()

# d) Izračunavanje točnosti, preciznosti i odziva na skupu podataka za testiranje
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Točnost:", accuracy)
print("Preciznost:", precision)
print("Odziv:", recall)

#Za drugi 5.5.1 b) i d)

#333333333333333333333333333333333333333333333333333333333333333333333333333333
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

# Učitavanje podataka
diabetes_df = pd.read_csv('pima-indians-diabetes.csv')

# Podjela na ulazne podatke X i izlazne podatke y
X = data.drop(columns=["Outcome"])
y = data["Outcome"] 

# Podjela podataka na skup za učenje i skup za testiranje (omjer 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# a) Izgradnja neuronske mreže
model = Sequential([
    Dense(12, input_dim=8, activation='relu'),  # Prvi skriveni sloj s 12 neurona i ReLU aktivacijom
    Dense(8, activation='relu'),                # Drugi skriveni sloj s 8 neurona i ReLU aktivacijom
    Dense(1, activation='sigmoid')              # Izlazni sloj s 1 neuron i sigmoid aktivacijom
])

# Ispis informacija o modelu
model.summary()

# b) Podešavanje procesa treniranja mreže
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# c) Pokretanje treniranja mreže
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=1)

# d) Pohrana modela na tvrdi disk
model.save("diabetes_model.h5")

# e) Evaluacija mreže na testnom skupu podataka
loss, accuracy = model.evaluate(X_test, y_test)
print("Točnost na testnom skupu podataka:", accuracy)

# f) Predikcija mreže na skupu podataka za testiranje
y_pred = model.predict(X_test)
########################################################################## Učitavanje dataset-a iz CSV datoteke
#KONV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Učitavanje podataka
diabetes_df = pd.read_csv('pima-indians-diabetes.csv')

# Podjela na ulazne podatke X i izlazne podatke y
X = diabetes_df.iloc[:, :-1].values
y = diabetes_df.iloc[:, -1].values

# Podjela podataka na skup za učenje i skup za testiranje (omjer 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prilagodba oblika ulaznih podataka za CNN
X_train = X_train.reshape(X_train.shape[0], 8, 1, 1)  # Očekivani oblik za CNN: (broj_primjera, visina, širina, dubina)
X_test = X_test.reshape(X_test.shape[0], 8, 1, 1)

# Izgradnja CNN modela
model = Sequential([
    Conv2D(32, kernel_size=(3, 1), activation='relu', input_shape=(8, 1, 1)),  # Konvolucijski sloj
    MaxPooling2D(pool_size=(2, 1)),                                            # Sloj za maksimalno grupiranje
    Flatten(),                                                                  # Ravnajući sloj
    Dense(128, activation='relu'),                                              # Potpuno povezani skriveni sloj
    Dense(1, activation='sigmoid')                                              # Izlazni sloj
])

# Podešavanje procesa treniranja CNN-a
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treniranje CNN-a
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# Evaluacija na testnom skupu podataka
loss, accuracy = model.evaluate(X_test, y_test)
print("Točnost na testnom skupu podataka:", accuracy)
##############################################################################################



# Učitavanje podataka
data = np.genfromtxt('pima-indians-diabetes.csv', delimiter=',')

# Podjela na ulazne podatke X i izlazne podatke y
X = data[:, :-1]  # Svi stupci osim posljednjeg
y = data[:, -1]   # Posljednji stupac

# Podjela podataka na skup za učenje i skup za testiranje (omjer 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Izgradnja modela logističke regresije
model = LogisticRegression()
model.fit(X_train, y_train)


data = pd.read_csv('pima-indians-diabetes.csv')

# Podjela podataka na ulazne značajke (X) i izlaznu varijablu (y)
X = data.drop(columns=['Outcome'])  # Uklanjanje stupca "Outcome" koji predstavlja izlaznu varijablu
y = data['Outcome']

# Podjela podataka na skup za učenje i skup za testiranje (omjer 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
                                                    

           