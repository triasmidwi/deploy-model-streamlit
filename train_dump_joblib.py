import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

#Load dataset
iris = load_iris()
X, y = iris.data, iris.target

#train a model
model = RandomForestClassifier()
model.fit(X,y)

#save the trained model
joblib.dump(model, 'model.joblib')