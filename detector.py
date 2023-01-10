from sklearn.datasets import load_iris
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


data = load_iris()
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.DataFrame(data=data.target, columns=["Class"])
classes = list(data.target_names)
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X, y)

pred_class = neigh.predict([[6.0, 3.4, 4.5, 1.6]])
print(classes[pred_class[0]])
