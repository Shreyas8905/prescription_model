import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
data = pd.read_csv("res2.csv")
X = data[["a1", "a2", "a3", "a4", "a5"]]  
y = data["res"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
dump(clf, "model.joblib")
print("✅ Model trained and saved as 'data.joblib'")
