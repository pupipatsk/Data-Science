from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series)
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4]
    }


df = df_terminal.copy()

# Apply ADF test to each column and store results in a new DataFrame
adf_results = {}
for column in df.columns:
    adf_results[column] = adf_test(df[column])

adf_results_df = pd.DataFrame(adf_results).T

# Display the ADF test results
adf_results_df.sort_values(by='p-value', ascending=False)