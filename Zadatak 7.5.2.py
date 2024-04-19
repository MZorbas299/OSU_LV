import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
J = []
plt.figure()
for i in range(1,10):
    km = KMeans(n_clusters=i, n_init=5)
    km.fit(img_array)
    labels = km.predict(img_array)
    J.append(km.inertia_)

plt.plot(range(1,10), J, '-o')
plt.title("Lakat")
img_array[:] = km.cluster_centers_[labels]

# rezultatna slika
img_array_aprox = img_array.copy()
plt.figure()
plt.imshow(img)
plt.show()