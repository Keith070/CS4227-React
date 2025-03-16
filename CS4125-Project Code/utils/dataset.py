import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataset(dataset_choice):
    if dataset_choice == '1':  # Gallery App Dataset
        return pd.read_csv('data/gallery_app.csv')
    elif dataset_choice == '2':  # Purchasing Emails Dataset
        return pd.read_csv('data/purchasing_data.csv')
    else:
        print("Invalid choice!")
        return None

def preprocess_data(df):
    df['cleaned_email_body'] = df['Mailbox'].str.lower().replace(r'[^a-z\s]', '', regex=True)
    df.dropna(subset=['cleaned_email_body'], inplace=True)  # Drop rows with null values
    return df

def vectorize_data(df):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['cleaned_email_body'])
    y = df['Type 2']  # Labels column
    return X, y
