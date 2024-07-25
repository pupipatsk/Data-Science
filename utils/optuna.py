# Hyperparameter tuning xgb with Optuna
import optuna
from sklearn.metrics import mean_squared_error


def objective(trial):
    """
    require: X_train, y_train, X_test, y_test
    """
    # Config search space
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }  # CatBoost

    model = XGBRegressor(**params, silent=True)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred_test = model.predict(X_test.to_numpy())
    rmse = mean_squared_error(y_test.to_numpy(), y_pred_test, squared=False)
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)
print("Best RMSE:", study.best_value)

# Visualize
# optuna.visualization.plot_intermediate_values(study)
# optuna.visualization.plot_contour(study)
optuna.visualization.plot_optimization_history(study)
