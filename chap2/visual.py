import optuna
import pandas

study = optuna.load_study(storage="sqlite:///optuna.db", study_name='ch2-adult_m')
df = study.trials_dataframe()

# optuna.visualization.plot_param_importances(
#     study=study,
#     params = ["gb_max_depth", "gb_min_samples_split"]
# ).show()

optuna.visualization.plot_contour(
    study=study,
    params = ["gb_max_depth", "gb_min_samples_split"]
).show()
