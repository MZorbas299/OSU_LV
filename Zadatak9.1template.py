import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
import tensorboard
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

#X_train_n = X_train_n[0:25000,:,:,:] #izdvajanje polovine skupa podataka za ucenje u svrhu zadatka



# 1-od-K kodiranje
y_train = to_categorical(y_train) #dtype ="uint8"
y_test = to_categorical(y_test) #dtype ="uint8"

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout (0.3)) #dodani dropout layer za 30% nasumicno iskljuci neurone
model.add(layers.Dense(10, activation='softmax'))

model.summary()


# Učitavanje log datoteke
#logdir = "logs/"

# Pokretanje Tensorboarda u notebooku
#notebook.start("--logdir " + logdir)


# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout',
                                update_freq = 100),
    keras.callbacks.EarlyStopping (monitor ="val_loss" ,patience = 5,verbose = 1),
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#optimizer = keras.optimizers.Adam() #konstruktoru predati proizvoljni learning_rate, postoji default
#model.compile(optimizer=optimizer,
                #loss='categorical_crossentropy',
                #metrics=['accuracy'])




model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')






#Zadatak 9.4.1 Skripta Zadatak_1.py ucitava CIFAR-10 skup podataka. Ovaj skup sadrži
#50000 slika u skupu za ucenje i 10000 slika za testiranje. Slike su RGB i rezolucije su 32x32.
#Svakoj slici je pridružena jedna od 10 klasa ovisno koji je objekt prikazan na slici. Potrebno je:

#1. Proucite dostupni kod. Od kojih se slojeva sastoji CNN mreža? Koliko ima parametara
#mreža?

#Sastoji se od 9 slojeva, dok sveukupno ima 1,122,758 parametara.

#2. Pokrenite ucenje mreže. Pratite proces ucenja pomocu alata Tensorboard na sljedeci nacin.
#Pokrenite Tensorboard u terminalu pomocu naredbe:
#tensorboard –logdir=logs
#i zatim otvorite adresu http://localhost:6006/ pomocu web preglednika.

#za tensorboard, otvoriti novi terminal (New terminal) i upisati tensorboard --logdir logs/cnn(_dropout, _earlyStop) 
#logs-direktorij u koji se sprema, naveden u Callbacku
#otvoriti dani link i prikazani su grafovi


#3. Proucite krivulje koje prikazuju tocnost klasifikacije i prosjecnu vrijednost funkcije gubitka
#na skupu podataka za ucenje i skupu podataka za validaciju. Što se dogodilo tijekom ucenja
#mreže? Zapišite tocnost koju ste postigli na skupu podataka za testiranje.

#Tocnost na testnom skupu podataka iznosi: 72.47
#Tijekom ucenja vidimo da nakon neke određene epohe, val_loss pocinje rasti te svim narednim epohama on raste




#Zadatak 9.4.2 Modificirajte skriptu iz prethodnog zadatka na nacin da na odgovarajuca mjesta u
#mrežu dodate droput slojeve. Prije pokretanja ucenja promijenite Tensorboard funkciju povratnog
#poziva na nacin da informacije zapisuje u novi direktorij (npr. =/log/cnn_droput). Pratite tijek
#ucenja. Kako komentirate utjecaj dropout slojeva na performanse mreže?



#Možemo vidjeti da dodavanjem dropout slojeva performasa mreže se poboljšala za par posto, dobili smo tocnost 75%, jer nasumičnim isključivanjem slojeva
#mreža bolje nauči te također val_loss nakon određene epohe(7) počinje rasti, no manje raste nego kada nismo imali dropout sloj, ali
#opet nije dobro da nam val_loss raste


#Zadatak 9.4.3 Dodajte funkciju povratnog poziva za rano zaustavljanje koja ce zaustaviti proces
#ucenja nakon što se 5 uzastopnih epoha ne smanji prosjecna vrijednost funkcije gubitka na
#validacijskom skupu.

#Na 5 epohi se val_loss krenuo povećavati, te se ucenje zaustavilo na 10 epohi, jer smo definirali ako se 5 uzastopnih epoha gubitak na validacijskom skupu
#povećava, neka se ucenje zaustavi.



#Zadatak 9.4.4 Što se dogada s procesom ucenja:
#1. ako se koristi jako velika ili jako mala velicina serije?

#2. ako koristite jako malu ili jako veliku vrijednost stope ucenja?

#3. ako izbacite odredene slojeve iz mreže kako biste dobili manju mrežu?

#4. ako za 50% smanjite velicinu skupa za ucenje?

#1 .jako velika velicina batcha - manje iteracija, krace trajanje epohe, losija tocnost i veci loss 
#1. jako mala velicina batcha - vise iteracija, duze trajanje epohe (predugo)
# 2. jako mala vrijednost stope ucenja - ucenje izrazito sporo konvergira, loss jedva pada, accuracy jedva raste 
#2. jako velika vrijednost stope ucenja - loss velik, accuracy mali, i ne mijenjaju se 
# 3. izbacivanje slojeva iz mreze za manju mrezu - dobijemo losiji rezultat
# 4. 50% manja velicina skupa za ucenje - upola manje iteracija, veci loss i losiji accuracy, epoha traje krace
