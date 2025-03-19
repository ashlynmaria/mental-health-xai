import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preparation import load_data

def preprocess_data(df):
    df['treatment'] = df['treatment'].apply(lambda x: 1 if x.strip().lower() == "yes" else 0)
    
    features = df.drop(columns=["treatment"])
    features = pd.get_dummies(features, drop_first=True)
    
    target = df['treatment']
    
    return features, target

def train_model():
    data = load_data()
    X, y = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, X_train, X_test

if __name__ == "__main__":
    train_model()
