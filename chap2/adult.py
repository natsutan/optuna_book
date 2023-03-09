import optuna

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

data = fetch_openml(name="adult")
X = pd.get_dummies(data["data"])
y = [1 if d == ">50K" else 0 for d in data["target"]]

def objective(trial):
    clf_name = trial.suggest_categorical("cls", ("RF", "GB"))

    if clf_name == "RF":
        clf = RandomForestClassifier(
            max_depth=trial.suggest_int("rf_max_depth", 2, 32,),
            min_samples_split=trial.suggest_float("rf_min_samples_split", 0, 1,)
        )
    else:
        clf = GradientBoostingClassifier(
            max_depth=trial.suggest_int("gb_max_depth", 2, 32,),
            min_samples_split = trial.suggest_float("gb_min_samples_split", 0, 1)
        )

    score = cross_val_score(clf, X, y, cv=3)
    accuracy = score.mean()
    return accuracy

stduy = optuna.create_study(direction="maximize", study_name="ch2-adult_m", storage="sqlite:///optuna.db")
stduy.optimize(objective, n_trials=100)

print(f"best acc {stduy.best_value}")
print(f"best param = {stduy.best_params}")

