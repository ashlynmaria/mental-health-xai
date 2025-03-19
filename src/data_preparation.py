import pandas as pd

def load_data(filepath='data/survey.csv'):
    data = pd.read_csv(filepath)
    print("Initial Data Info:")
    print(data.info())
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna("Unknown")
    data = data.drop(columns=["Timestamp", "comments"])
    
    return data

if __name__ == "__main__":
    df = load_data()
    print("Data head:")
    print(df.head())
