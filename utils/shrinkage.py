import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler


# Regression Coefficients Shrinkage

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

alphas = np.logspace(-1, 7, 100) # * config space

coefs = []

for alpha in alphas:
    model = Lasso(alpha=alpha)
    model = Ridge(alpha=alpha)
    
    model.fit(X_scaled, y)
    coefs.append(model.coef_)

# Plot
plt.figure(figsize=(16, 9))
for i in range(10):
    plt.plot(alphas, [coef[i] for coef in coefs], label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Regression Coefficients Shrinkage')
plt.legend()
plt.grid(True)
plt.show()
