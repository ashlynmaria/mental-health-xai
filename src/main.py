from model_training import train_model
from explainability import explain_model

def main():
    model, X_train, X_test = train_model()
    explanation = explain_model(model, X_train, X_test)
    print("Model training and explanation complete.")

if __name__ == "__main__":
    main()
