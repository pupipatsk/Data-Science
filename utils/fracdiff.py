from fracdiff.sklearn import FracdiffStat


f = FracdiffStat(window=12, upper=10)
X = f.fit_transform(df_terminal) # 2d time-series with shape (n_samples, n_features)

X = pd.DataFrame(X, index=df_terminal.index, columns=df_terminal.columns)
df_terminal = X

f.d_