import pandas as pd

df = pd.read_csv('C:/Users/Usuario/Desktop/experimentacion/archivos python/heart.csv')

import numpy as np
from sklearn import metrics

from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectKBest

X = df.drop("target", axis=1)
y = df["target"]

#Secciones empleando extra trees y mutual information en cada tecnica

#----------------------
#REGRESION LOGISTICA 
#----------------------

#--Extra Trees
selector = SelectFromModel(
    ExtraTreesClassifier(
        n_estimators=100,
        random_state=42
    ),
    threshold=-float("inf")
)

pipe = Pipeline([
    ("selector", selector),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000,random_state=42))
])

param_grid = {
    "selector__max_features": list(range(1, X.shape[1] + 1)),
    "logreg__C": [0.01, 0.1, 1, 10],
    "logreg__solver": ["liblinear", "lbfgs"],
    "logreg__penalty": ["l2"]
}
#--

#--Mutual information
discrete_features = [False,  True,   True,   False,  False,  True,   True,   
                     False,  True,   False,  True,   True,   True]

def mutual_info_score(X, y):
    return mutual_info_classif(X,y,discrete_features=discrete_features,random_state=42)

pipe = Pipeline([
    ("selector", SelectKBest(score_func=mutual_info_score)),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000,random_state=42))
])

param_grid = {
    "selector__k": list(range(1, X.shape[1] + 1)),
    "logreg__C": [0.01, 0.1, 1, 10],
    "logreg__solver": ["liblinear", "lbfgs"],
    "logreg__penalty": ["l2"]
}
#--

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="recall", n_jobs=-1)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(grid, X, y, cv=outer_cv,
                         return_estimator=True, scoring=["accuracy", "precision", "recall", "f1"], return_train_score=True,
                         n_jobs=-1)

print("\n" + "=" * 70)
print("Mejores hiperparámetros y variables seleccionadas por fold:")
print("=" * 70)
for i, est in enumerate(results["estimator"]):

    best_pipeline = est.best_estimator_
    selector = best_pipeline.named_steps["selector"]
    selected_features = X.columns[selector.get_support()]

    print(f"\n========== OUTER FOLD {i+1} ==========")

    print("\n--- INNER CV ---")
    print("Variables seleccionadas:", list(selected_features))
    print("Número de variables:", len(selected_features))
    print("Mejores hiperparámetros:", est.best_params_)
    print("Mejor recall interno:", est.best_score_) 

    print("\n--- OUTER CV ---")
    print("Accuracy:", results["test_accuracy"][i])
    print("Precision:", results["test_precision"][i])
    print("Recall:", results["test_recall"][i]) 
    print("F1:", results["test_f1"][i])

    print("\n--- ENTRENAMIENTO ---")
    print("Accuracy:", results["train_accuracy"][i])
    print("Precision:", results["train_precision"][i])
    print("Recall:", results["train_recall"][i]) 
    print("F1:", results["train_f1"][i])

print("\n" + "=" * 70)
print("RESULTADOS FINALES:")
print("=" * 70)

print("Accuracy:", results["test_accuracy"].mean())
print("Precision:", results["test_precision"].mean())
print("Recall:", results["test_recall"].mean())
print("F1:", results["test_f1"].mean())

#----------------------
#NAIVE BAYES
#----------------------

#--Extra Trees
selector = SelectFromModel(
    ExtraTreesClassifier(n_estimators=100,random_state=42),
    threshold=-float("inf")
)

pipe = Pipeline([
    ("selector", selector),
    ("scaler", StandardScaler()),
    ("nb", GaussianNB())
])

param_grid = {
    "selector__max_features": list(range(1, X.shape[1] + 1)),
    "nb__var_smoothing": np.logspace(-12, -6, 7) 
}
#--

#--Mutual information
discrete_features = [False,  True,   True,   False,  False,  True,   True,   
                     False,  True,   False,  True,   True,   True]

def mutual_info_score(X, y):
    return mutual_info_classif(X,y,discrete_features=discrete_features,random_state=42)

pipe = Pipeline([
    ("selector", SelectKBest(score_func=mutual_info_score)),
    ("scaler", StandardScaler()),
    ("nb", GaussianNB())
])

param_grid = {
    "selector__k": list(range(1, X.shape[1] + 1)),
    "nb__var_smoothing": np.logspace(-12, -6, 7) 
}
#--

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="recall", n_jobs=-1)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = cross_validate(grid, X, y, cv=outer_cv,
                         return_estimator=True, scoring=["accuracy", "precision", "recall", "f1"], return_train_score=True,
                         n_jobs=-1)

print("\n" + "=" * 70)
print("Mejores hiperparámetros y variables seleccionadas por fold:")
print("=" * 70)
for i, est in enumerate(results["estimator"]):

    best_pipeline = est.best_estimator_
    selector = best_pipeline.named_steps["selector"]
    selected_features = X.columns[selector.get_support()]

    print(f"\n========== OUTER FOLD {i+1} ==========")

    print("\n--- INNER CV ---")
    print("Variables seleccionadas:", list(selected_features))
    print("Número de variables:", len(selected_features))
    print("Mejores hiperparámetros:", est.best_params_)
    print("Mejor recall interno:", est.best_score_) 

    print("\n--- OUTER CV ---")
    print("Accuracy:", results["test_accuracy"][i])
    print("Precision:", results["test_precision"][i])
    print("Recall:", results["test_recall"][i]) 
    print("F1:", results["test_f1"][i])

    print("\n--- ENTRENAMIENTO ---")
    print("Accuracy:", results["train_accuracy"][i])
    print("Precision:", results["train_precision"][i])
    print("Recall:", results["train_recall"][i]) 
    print("F1:", results["train_f1"][i])

print("\n" + "=" * 70)
print("RESULTADOS FINALES:")
print("=" * 70)

print("Accuracy:", results["test_accuracy"].mean())
print("Precision:", results["test_precision"].mean())
print("Recall:", results["test_recall"].mean())
print("F1:", results["test_f1"].mean())

#----------------------
#SVM
#----------------------

#--Extra Trees
selector = SelectFromModel(
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    threshold=-float("inf")
)

pipe = Pipeline([
    ("selector", selector),
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

param_grid = {
    "selector__max_features": list(range(1, X.shape[1] + 1)),
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", "auto"]
}
##--

#--Mutual information
discrete_features = [False,  True,   True,   False,  False,  True,   True,   
                     False,  True,   False,  True,   True,   True]

def mutual_info_score(X, y):
    return mutual_info_classif(X,y,discrete_features=discrete_features,random_state=42)

pipe = Pipeline([
    ("selector", SelectKBest(score_func=mutual_info_score)),
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

param_grid = {
    "selector__k": list(range(1, X.shape[1] + 1)),
    "svm__C": [0.1, 1, 10],            
    "svm__kernel": ["linear", "rbf"], 
    "svm__gamma": ["scale", "auto"]    
}
##--

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

grid = GridSearchCV(pipe, param_grid, cv=inner_cv,
                    scoring="recall", n_jobs=-1)

results = cross_validate(grid, X, y, cv=outer_cv,
                         return_estimator=True, scoring=["accuracy", "precision", "recall", "f1"], return_train_score=True,
                         n_jobs=-1)

print("\n" + "=" * 70)
print("Mejores hiperparámetros y variables seleccionadas por fold:")
print("=" * 70)
for i, est in enumerate(results["estimator"]):

    best_pipeline = est.best_estimator_
    selector = best_pipeline.named_steps["selector"]
    selected_features = X.columns[selector.get_support()]

    print(f"\n========== OUTER FOLD {i+1} ==========")

    print("\n--- INNER CV ---")
    print("Variables seleccionadas:", list(selected_features))
    print("Número de variables:", len(selected_features))
    print("Mejores hiperparámetros:", est.best_params_)
    print("Mejor recall interno:", est.best_score_) 

    print("\n--- OUTER CV ---")
    print("Accuracy:", results["test_accuracy"][i])
    print("Precision:", results["test_precision"][i])
    print("Recall:", results["test_recall"][i]) 
    print("F1:", results["test_f1"][i])

    print("\n--- ENTRENAMIENTO ---")
    print("Accuracy:", results["train_accuracy"][i])
    print("Precision:", results["train_precision"][i])
    print("Recall:", results["train_recall"][i]) 
    print("F1:", results["train_f1"][i])

print("\n" + "=" * 70)
print("RESULTADOS FINALES:")
print("=" * 70)

print("Accuracy:", results["test_accuracy"].mean())
print("Precision:", results["test_precision"].mean())
print("Recall:", results["test_recall"].mean())
print("F1:", results["test_f1"].mean())

#----------------------
#RNADOM FOREST
#----------------------

#--Extra Trees
selector = SelectFromModel(
    ExtraTreesClassifier(n_estimators=100,random_state=42),
    threshold=-float("inf") 
)

pipe = Pipeline([
    ("selector", selector),
    ("rf", RandomForestClassifier(random_state=42,n_jobs=-1))
])


param_grid = {
    "selector__max_features": list(range(1, X.shape[1] + 1)),
    "rf__n_estimators": [100, 200, 300],       
    "rf__min_samples_split": [2, 5, 10],       
    "rf__max_features": ["sqrt", "log2", None] 
}
#--

#--Mutual information
discrete_features = [False,  True,   True,   False,  False,  True,   True,   
                     False,  True,   False,  True,   True,   True]

def mutual_info_score(X, y):
    return mutual_info_classif(X,y,discrete_features=discrete_features,random_state=42)

pipe = Pipeline([
    ("selector", SelectKBest(score_func=mutual_info_score)),
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

param_grid = {
    "selector__k": list(range(1, X.shape[1] + 1)),
    "rf__n_estimators": [100, 200, 300],        
    "rf__min_samples_split": [2, 5, 10],         
    "rf__max_features": ["sqrt", "log2", None] 
}
#--

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="recall", n_jobs=-1)

results = cross_validate(grid, X, y, cv=outer_cv,
                         return_estimator=True, scoring=["accuracy", "precision", "recall", "f1"], return_train_score=True,
                         n_jobs=-1)

print("\n" + "=" * 70)
print("Mejores hiperparámetros y variables seleccionadas por fold:")
print("=" * 70)
for i, est in enumerate(results["estimator"]):

    best_pipeline = est.best_estimator_
    selector = best_pipeline.named_steps["selector"]
    selected_features = X.columns[selector.get_support()]

    print(f"\n========== OUTER FOLD {i+1} ==========")

    print("\n--- INNER CV ---")
    print("Variables seleccionadas:", list(selected_features))
    print("Número de variables:", len(selected_features))
    print("Mejores hiperparámetros:", est.best_params_)
    print("Mejor recall interno:", est.best_score_) 

    print("\n--- OUTER CV ---")
    print("Accuracy:", results["test_accuracy"][i])
    print("Precision:", results["test_precision"][i])
    print("Recall:", results["test_recall"][i]) 
    print("F1:", results["test_f1"][i])

    print("\n--- ENTRENAMIENTO ---")
    print("Accuracy:", results["train_accuracy"][i])
    print("Precision:", results["train_precision"][i])
    print("Recall:", results["train_recall"][i]) 
    print("F1:", results["train_f1"][i])

print("\n" + "=" * 70)
print("RESULTADOS FINALES:")
print("=" * 70)

print("Accuracy:", results["test_accuracy"].mean())
print("Precision:", results["test_precision"].mean())
print("Recall:", results["test_recall"].mean())
print("F1:", results["test_f1"].mean())
