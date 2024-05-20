import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()

# 1.1

# Model KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_n, y_train)
y_train_p = knn.predict(X_train)
y_test_p = knn.predict(X_test)

print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

plot_decision_regions(X_train_n, y_train, classifier=knn)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("knn5")
plt.tight_layout()

# 1.2

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train_n, y_train)
y_train_p = knn1.predict(X_train)
y_test_p = knn1.predict(X_test)

plot_decision_regions(X_train_n, y_train, classifier=knn1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("knn1")
plt.tight_layout()


knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(X_train_n, y_train)
y_train_p = knn100.predict(X_train)
y_test_p = knn100.predict(X_test)

plot_decision_regions(X_train_n, y_train, classifier=knn100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("knn100")
plt.tight_layout()

# 2 zad

scores = cross_val_score(knn, X_train_n ,y_train ,cv=5)
print(scores)
# Za k = 10 mi daje dobre rezultate, bolje nego k=5

param_grid = {'n_neighbors': np.arange(1, 101)}
svm_gscv = GridSearchCV ( estimator= KNeighborsClassifier().fit(X_train_n, y_train) , param_grid=param_grid , cv=5 , scoring='accuracy')
svm_gscv.fit(X_train_n, y_train)
print ( svm_gscv.best_params_ )
#print ( svm_gscv.best_score_ )
#print ( svm_gscv.cv_results_ )

# 3 zad

SVM_model=svm.SVC(kernel='poly', gamma = 1 , C=1)
SVM_model.fit(X_train_n , y_train)
y_test_p_SVM = SVM_model.predict(X_train)
y_test_p_SVM = SVM_model.predict(X_test)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("SVM")
plt.tight_layout()

# zad 4
param_grid = {'C': [10 , 10, 100 , 100 ], 'gamma': [10 , 1 , 0.1 , 0.01 ]}
svm_gscv = GridSearchCV ( estimator= svm.SVC().fit(X_train_n, y_train) , param_grid=param_grid , cv=5 , scoring='accuracy')
svm_gscv.fit(X_train_n, y_train)
print ( svm_gscv.best_params_ )
#print ( svm_gscv.best_score_ )
#print ( svm_gscv.cv_results_ )


plt.show()

/////////////////////////////////////////////////////////////////////////////////////////////

# Inicijalizacija liste za pohranu vrijednosti sume kvadratnih udaljenosti
sse = []

# Testiranje broja klastera od 1 do 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(iris.data)
    sse.append(kmeans.inertia_)

# Prikazivanje rezultata pomoću grafa lakta
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Broj klastera (K)')
plt.ylabel('Suma kvadratnih udaljenosti (SSE)')
plt.title('Elbow Method za pronalaženje optimalnog broja klastera')
plt.show()

optimalni_broj_klastera = 3  # Pretpostavljamo da smo dobili optimalni broj klastera
kmeans = KMeans(n_clusters=optimalni_broj_klastera)
kmeans.fit(iris.data)

plt.scatter(iris.data[:, 0], iris.data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=300, c='black', label='Centroidi')
plt.xlabel('Duljina latice (cm)')
plt.ylabel('Duljina čašice (cm)')
plt.title('Klasteri cvijeta irisa')
plt.legend()
plt.show()

# Stvarne oznake
stvarne_oznake = iris.target

# Izračunane oznake klastiranja
oznake_klastiranja = kmeans.labels_

# Izračun preciznosti klasifikacije
preciznost = accuracy_score(stvarne_oznake, oznake_klastiranja)
print("Preciznost klasifikacije: {:.2f}".format(preciznost))