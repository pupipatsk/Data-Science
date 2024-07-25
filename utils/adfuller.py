from statsmodels.tsa.stattools import adfuller

# Stationary
# - p-value < 0.05
# - ADF Statistic < Critical Values
# 
# """explaination
# If the test statistic is less than the critical value 
# or if the p-value is less than a pre-specified significance level (e.g., 0.05), 
# then the null hypothesis is rejected and the time series is considered stationary.
# """

def adf_test(series):
    result = adfuller(series)
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4]
    }


# * Data Terminal
df = df_terminal.copy()

# # Apply ADF to Series
# adf_test(df['column_name'])

# Apply ADF to DataFrame
adf_results = {}
for column in df.columns:
    adf_results[column] = adf_test(df[column])
adf_results_df = pd.DataFrame(adf_results).T # convert to DataFrame
# results
adf_results_df.sort_values(by='p-value', ascending=False)
