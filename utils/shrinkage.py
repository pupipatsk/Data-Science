# LinearModel/Regression Coefficients Shrinkage
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt

# req. input data: X, y
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# * config space
alphas = np.logspace(-1, 7, 100)

coefs = []
for alpha in alphas:
    # choose model
    model = Lasso(alpha=alpha)
    model = Ridge(alpha=alpha)

    model.fit(X_scaled, y)
    coefs.append(model.coef_)

# Plot
plt.figure(figsize=(16, 9))
for i in range(10):
    plt.plot(alphas, [coef[i] for coef in coefs], label=f"Feature {i+1}")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.title("Regression Coefficients Shrinkage")
plt.legend()
plt.grid(True)
plt.show()


# --- Linear model Coef. --- #

print(model)
sumcoef = sum(abs(model.coef_))
df_coef = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
df_coef["Weight (%)"] = abs(df_coef["Coefficient"]) / sumcoef * 100
df_coef["Sign"] = np.where(df_coef["Coefficient"] > 0, "+", "-")
df_coef = df_coef[["Feature", "Sign", "Weight (%)", "Coefficient"]]  # rearrange columns
df_coef = df_coef.sort_values(by="Weight (%)", ascending=False)
print(model.alpha_)
print(df_coef.to_string())
