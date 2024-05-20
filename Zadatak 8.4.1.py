from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Učitavanje Iris dataset-a
data = pd.read_csv("winequality-red.csv", sep=";")

# Zamjena vrijednosti kvalitete vina manje od 6 s 0, a one koje imaju vrijednost 6 ili veću s 1
data['quality'] = data['quality'].apply(lambda x: 0 if x < 6 else 1)
# Podjela podataka na ulazne podatke X i izlazne podatke y
X = data.drop('quality', axis=1)
y = data['quality']

# Podjela podataka na skup za učenje i skup za testiranje (omjer 75:25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizacija podataka - skaliranje značajki
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kreiranje modela
model = Sequential()

# Dodavanje prvog skrivenog sloja
model.add(Input(shape=(X_train.shape[1],)))

# Dodavanje Dropout sloja između prvog i drugog sloja
model.add(Dense(22, activation="relu"))

# Dodavanje drugog skrivenog sloja
model.add(Dense(12, activation='relu'))

# Dodavanje Dropout sloja između drugog i trećeg sloja

# Dodavanje trećeg skrivenog sloja
model.add(Dense(4, activation='relu'))

# Dodavanje izlaznog sloja
model.add(Dense(1, activation='sigmoid'))

# Ispis informacija o mreži
model.summary()

# Kompilacija modela
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Treniranje modela
# Kodiranje ciljnih podataka u one-hot vektore
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
history = model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=7, validation_data=(X_test_scaled, y_test_encoded))

# Pohranjivanje modela na tvrdi disk
model.save("iris_model.h5")

# Učitavanje modela s diska
loaded_model = load_model("iris_model.h5")

# Evaluacija modela na testnom skupu
loss, accuracy = loaded_model.evaluate(X_test_scaled, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predikcija na skupu podataka za testiranje
y_pred = loaded_model.predict(X_test_scaled)

# Pretvaranje predikcija u indekse klasa


# Izračunavanje matrice zabune
conf_matrix = confusion_matrix(y_test_encoded, y_pred.round())

# Vizualizacija matrice zabune
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

//////////////////////////////////////////////////////////////////////////////////


# Učitavanje Iris dataset-a
from sklearn import datasets
iris = datasets.load_iris()

# Ulazni podaci (X) su značajke, a izlazni podaci (y) su ciljne vrijednosti (klase cvijeta)
X = iris.data
y = iris.target

# Podjela podataka na skup za učenje i skup za testiranje (omjer 75:25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardizacija podataka - skaliranje značajki
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kreiranje modela
model = Sequential()

# Dodavanje prvog skrivenog sloja
model.add(Input(shape=(X_train.shape[1],)))

# Dodavanje Dropout sloja između prvog i drugog sloja
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.3))

# Dodavanje drugog skrivenog sloja
model.add(Dense(7, activation='relu'))

# Dodavanje Dropout sloja između drugog i trećeg sloja
model.add(Dropout(0.3))

# Dodavanje trećeg skrivenog sloja
model.add(Dense(5, activation='relu'))

# Dodavanje izlaznog sloja
model.add(Dense(3, activation='softmax'))

# Ispis informacija o mreži
model.summary()

# Kompilacija modela
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Treniranje modela
# Kodiranje ciljnih podataka u one-hot vektore
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
history = model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=7, validation_data=(X_test_scaled, y_test_encoded))

# Pohranjivanje modela na tvrdi disk
model.save("iris_model.h5")

# Učitavanje modela s diska
loaded_model = load_model("iris_model.h5")

# Evaluacija modela na testnom skupu
loss, accuracy = loaded_model.evaluate(X_test_scaled, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predikcija na skupu podataka za testiranje
y_pred = loaded_model.predict(X_test_scaled)

# Pretvaranje predikcija u indekse klasa
y_pred_classes = np.argmax(y_pred, axis=1)

# Pretvaranje stvarnih oznaka u indekse klasa
y_test_classes = np.argmax(y_test_encoded, axis=1)

# Izračunavanje matrice zabune
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Vizualizacija matrice zabune
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()