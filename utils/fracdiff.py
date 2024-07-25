from fracdiff.sklearn import FracdiffStat


df = df_terminal.copy()

f = FracdiffStat(window=12, upper=10)
X = f.fit_transform(df)  # 2d time-series with shape (n_samples, n_features)

X = pd.DataFrame(X, index=df.index, columns=df.columns)

df_terminal = X  # update
del df, X

f.d_
