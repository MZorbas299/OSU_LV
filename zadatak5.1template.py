import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . linear_model import LogisticRegression
from sklearn . metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay, accuracy_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# a)
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='x')

# b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c)
plt.plot(X_train[:,1], -LogRegression_model.coef_[0, 1]/LogRegression_model.coef_[0, 0]*X_train[:,1] - LogRegression_model.intercept_[0]*LogRegression_model.coef_[0, 0])

# d)
y_test_p = LogRegression_model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test ,y_test_p ))
disp.plot()
print ("Tocnost:" , accuracy_score(y_test ,y_test_p))
print (classification_report(y_test ,y_test_p))
# e)
plt.figure()
y_correct = y_test_p==y_test
print(y_correct)
print(X_test[:,0])
plt.scatter(X_test[:,0], X_test[:,1], c=y_correct , cmap='Set1')

plt.show()