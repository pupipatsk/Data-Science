import time
from sklearn.metrics import mean_squared_error, r2_score


def modelling(model):
    """
    require: X_train, y_train, X_test, y_test
    """
    start = time.time()
    print(f"Start training: {model}")

    print("Fitting...")
    start_fitting = time.time()
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    end_fitting = time.time()
    print(f"Fitting time: {end_fitting-start_fitting :.2f} seconds")

    print("Predicting...")
    y_pred_train = model.predict(X_train)
    y_pred_train = pd.Series(y_pred_train, index=y_train.index)

    y_pred_test = model.predict(X_test)
    y_pred_test = pd.Series(y_pred_test, index=y_test.index)

    end = time.time()
    print(f"Finished, Training time: {end-start :.2f} seconds")

    return y_pred_train, y_pred_test


def evaluation(y_train, y_pred_train, y_test, y_pred_test):
    """
    evaluate car price prediction
    in both Train and Test set

    metrics
    - MAE
    - MAPE
    - RMSE
    - R-squared
    """

    # MAE
    mae_train = (abs(y_train - y_pred_train)).mean()
    mae_test = (abs(y_test - y_pred_test)).mean()
    # MAPE
    mape_train = (abs(y_train - y_pred_train) / y_train).mean() * 100
    mape_test = (abs(y_test - y_pred_test) / y_test).mean() * 100
    # RMSE
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    # R-squared
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Printing
    # print(f"MAE Train: {mae_train :,.2f}")
    print(f"MAPE Test: {mape_test :.2f} %")
    print(f"RMSE Test: {rmse_test :,.2f}")

    # print(f"MAE Test: {mae_test :,.2f}")
    print(f"MAPE Train: {mape_train :.2f} %")
    print(f"RMSE Train: {rmse_train :,.2f}")

    print(f"R-squared Test: {r2_test :.4f}")
    print(f"R-squared Train: {r2_train :.4f}")

    # result dataframe
    df_result = pd.DataFrame(
        {
            "Train": [mae_train, mape_train, rmse_train, r2_train],
            "Test": [mae_test, mape_test, rmse_test, r2_test],
        },
        index=["MAE", "MAPE", "RMSE", "R2"],
    )
    df_result["Diff(%)"] = (
        (df_result["Test"] - df_result["Train"]) / df_result["Train"] * 100
    )
    return df_result
