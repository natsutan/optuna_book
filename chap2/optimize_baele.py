import optuna

def objective(trial):
    x = trial.suggest_float("x", -4.5, 4.5)
    y = trial.suggest_float("y", -4.5, 4.5)

    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2 ) ** 2 + (2.625 - x + x * y ** 3) ** 2

stduy = optuna.create_study(direction='minimize')
stduy.optimize(objective, n_trials=1000)

print(f"Best objective value: {stduy.best_value}")
print(f"Best parameter: {stduy.best_params}")

