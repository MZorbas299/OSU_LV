import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=True)

print(data[0:5,])

print("Broj osoba: " + str(len(data)))

plt.scatter(data[0::50, 2], data[0::50, 1])
plt.title("Odnos visine i tezine osobe")
plt.xlabel("Tezina")
plt.ylabel("Visina")

print("Min:" + str(data[0::50, 1].min()))
print("Max:" + str(data[0::50, 1].max()))
print("Mean:" + str(data[0::50, 1].mean()))


indm = (data[:,0] == 1)

print("Min M:" + str(data[indm, 1].min()))
print("Max M:" + str(data[indm, 1].max()))
print("Mean M:" + str(data[indm, 1].mean()))

indZ = (data[:,0] == 0)

print("Min Z:" + str(data[indZ, 1].min()))
print("Max Z:" + str(data[indZ, 1].max()))
print("Mean Z:" + str(data[indZ, 1].mean()))

plt.show()

