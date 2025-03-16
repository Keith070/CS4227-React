from config_manager import ConfigurationManager  # Import the ConfigurationManager
from classifiers.RandomForestClassifier import RandomForestClassifierModel
from classifiers.SVMClassifier import SVMClassifierModel
from classifiers.AdaBoostClassifier import AdaBoostClassifierModel
from classifiers.NaiveBayesClassifier import NaiveBayesClassifierModel
from classifiers.LogisticRegressionClassifier import LogisticRegressionClassifierModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from observer import Subject, LoggingObserver  # Import missing classes

class ClassifierFacade:
    def __init__(self):
        # Initialize subject and logging observer for the observer pattern
        self.subject = Subject()
        
        # Use ConfigurationManager to get the log file path
        config = ConfigurationManager()  # Get config
        self.logging_observer = LoggingObserver(log_file=config.logging_file)  # Pass log file path
        self.subject.attach(self.logging_observer)

    # Load the dataset based on choice
    def load_dataset(self, dataset_choice):
        if dataset_choice == '1':
            return pd.read_csv('data/gallery_app.csv')
        elif dataset_choice == '2':
            return pd.read_csv('data/purchasing_data.csv')
        else:
            raise ValueError("Invalid dataset choice!")

    # Preprocess the dataset (cleaning and transforming the email body)
    def preprocess_data(self, df):
        df['cleaned_email_body'] = df['Mailbox'].str.lower().replace(r'[^a-z\s]', '', regex=True)
        return df

    # Vectorize the dataset (convert email body text to numerical features)
    def vectorize_data(self, df):
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(df['cleaned_email_body'])
        y = df['Type 2']
        return X, y

    # Choose classifier, tune hyperparameters, and return the best model
    def choose_classifier(self, classifier_choice, X_train, y_train):
        param_grid = {}
        if classifier_choice == 'random_forest':
            model = RandomForestClassifierModel()
            param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        elif classifier_choice == 'svm':
            model = SVMClassifierModel()
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif classifier_choice == 'adaboost':
            model = AdaBoostClassifierModel()
            param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1]}
        elif classifier_choice == 'naive_bayes':
            model = MultinomialNB()
            param_grid = {'alpha': [0.1, 0.5, 1.0]}
        elif classifier_choice == 'logistic_regression':
            model = LogisticRegressionClassifierModel()
            param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        else:
            raise ValueError("Invalid classifier choice.")

        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    # Evaluate the model and print results (accuracy and confusion matrix)
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Notify observers (logging accuracy and confusion matrix)
        self.subject.notify({
            "accuracy": f"{accuracy * 100:.2f}%",
            "confusion_matrix": conf_matrix.tolist()
        })
        return accuracy, conf_matrix
