import shap
from model_training import train_model

def explain_model(model, X_train, X_test, sample_index=0):
    explainer = shap.KernelExplainer(model.predict_proba, X_train, link="logit")
    
    shap_values = explainer.shap_values(X_test.iloc[[sample_index]])
    
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[sample_index, :])
    return force_plot

if __name__ == "__main__":
    model, X_train, X_test = train_model()
    explanation = explain_model(model, X_train, X_test)
    shap.save_html("explanation.html", explanation)
    print("Explanation saved as explanation.html")
