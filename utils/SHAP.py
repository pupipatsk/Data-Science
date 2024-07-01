import shap
shap.initjs()


X_sampled = X_train.sample(100, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sampled.to_numpy())

shap.force_plot(explainer.expected_value, shap_values[0,:], X_sampled.iloc[0,:])

shap.force_plot(explainer.expected_value, shap_values, X_sampled)

shap.summary_plot(shap_values, X_sampled)

shap.summary_plot(shap_values, X_sampled, plot_type="bar")

for name in X_sampled.columns:
    shap.dependence_plot(name, shap_values, X_sampled, display_features=X_sampled)