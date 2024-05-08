import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
figure, axis = plt.subplots(2, 2)
axis[0][0].imshow(x_train[13])
axis[0][1].imshow(x_train[143])
axis[1][0].imshow(x_train[22])
axis[1][1].imshow(x_train[31])
# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

x_train_s = np.reshape(x_train_s, (60000, 28*28))
x_test_s = np.reshape(x_test_s, (10000, 28*28))


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape = (28*28, )))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
print(model.summary())
# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy",])


# TODO: provedi ucenje mreze

batch_size = 32
epochs = 20
history = model.fit(x_train_s, y_train_s, batch_size = batch_size, epochs = epochs, validation_split=0.1)


# TODO: Prikazi test accuracy i matricu zabune

y_pred = model.predict(x_test_s)
score = model.evaluate(x_test_s, y_test_s, verbose=0 )

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, np.argmax(y_pred, axis=1)))
disp.plot()
plt.show()

# TODO: spremi model

model.save("FCN/")
del model