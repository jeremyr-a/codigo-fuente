import pandas as pd

df = pd.read_csv('C:/Users/Usuario/Desktop/experimentacion/archivos python/heart.csv')

import numpy as np
from sklearn import metrics

from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

X = df.iloc[:, [2, 9, 11, 12]].values
y = df.iloc[:,-1].values

#REGRESION LOGISTICA

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    "logreg__C": [0.01, 0.1, 1, 10],
    "logreg__solver": ["liblinear", "lbfgs"],
    "logreg__penalty": ["l2"]
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="recall", n_jobs=-1)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)

results = cross_validate(
    grid, X, y, cv=outer_cv,
    scoring="recall",
    return_estimator=True,
    n_jobs=-1
)

print("\nMejores hiperparámetros por fold externo:")
for i, est in enumerate(results["estimator"]):
    print(f"Fold {i+1}: {est.best_params_}")


print("Matriz de confusión:\n", confusion_matrix(y, y_pred))
print("\nReporte de clasificación:\n", classification_report(y, y_pred))

exactitud = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
sensibilidad = recall_score(y, y_pred)
puntaje = f1_score(y, y_pred)

print("EXACTITUD:", exactitud)
print("PRECISION:", precision)
print("SENSIBILIDAD:", sensibilidad)
print("PUNTAJE DE F1:", puntaje)

# NAIVE BAYES

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("nb", GaussianNB())
])

param_grid = {
    "nb__var_smoothing": np.logspace(-12, -6, 7) 
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="recall", n_jobs=-1)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)

results = cross_validate(
    grid, X, y, cv=outer_cv,
    scoring="recall",
    return_estimator=True,
    n_jobs=-1
)

print("\nMejores hiperparámetros por fold externo:")
for i, est in enumerate(results["estimator"]):
    print(f"Fold {i+1}: {est.best_params_}")

print("Matriz de confusión:\n", confusion_matrix(y, y_pred))
print("\nReporte de clasificación:\n", classification_report(y, y_pred))

exactitud = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
sensibilidad = recall_score(y, y_pred)
puntaje = f1_score(y, y_pred)

print("EXACTITUD:", exactitud)
print("PRECISION:", precision)
print("SENSIBILIDAD:", sensibilidad)
print("PUNTAJE DE F1:", puntaje)

# SVM

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

param_grid = {
    "svm__C": [0.1, 1, 10],            
    "svm__kernel": ["linear", "rbf"], 
    "svm__gamma": ["scale", "auto"]    
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

grid = GridSearchCV(pipe, param_grid, cv=inner_cv,
                    scoring="recall", n_jobs=-1)

y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)

print("Matriz de confusión:\n", confusion_matrix(y, y_pred))
print("\nReporte de clasificación:\n", classification_report(y, y_pred))

results = cross_validate(grid, X, y, cv=outer_cv,
                         return_estimator=True, scoring="recall", n_jobs=-1)

print("\nMejores hiperparámetros por fold externo:")
for i, est in enumerate(results["estimator"]):
    print(f"Fold {i+1}: {est.best_params_}")
    
exactitud = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
sensibilidad = recall_score(y, y_pred)
puntaje = f1_score(y, y_pred)

print("EXACTITUD:", exactitud)
print("PRECISION:", precision)
print("SENSIBILIDAD:", sensibilidad)
print("PUNTAJE DE F1:", puntaje)

#RANDOM FOREST
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "rf__n_estimators": [100, 200, 300],       
    "rf__max_depth": [None, 5, 10, 20],        
    "rf__min_samples_split": [2, 5, 10],       
    "rf__min_samples_leaf": [1, 2, 4],         
    "rf__max_features": ["sqrt", "log2", None] 
}

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="recall", n_jobs=-1)

y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)

results = cross_validate (
    grid, X, y, cv=outer_cv,
    scoring="recall",
    return_estimator=True,
    n_jobs=-1
)

print("\nMejores hiperparámetros por fold externo:")
for i, est in enumerate(results["estimator"]):
    print(f"Fold {i+1}: {est.best_params_}")

print("\nMatriz de confusión:\n", confusion_matrix(y, y_pred))
print("\nReporte de clasificación:\n", classification_report(y, y_pred))

exactitud = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
sensibilidad = recall_score(y, y_pred)
puntaje = f1_score(y, y_pred)

print("EXACTITUD:", exactitud)
print("PRECISION:", precision)
print("SENSIBILIDAD:", sensibilidad)
print("PUNTAJE DE F1:", puntaje)
