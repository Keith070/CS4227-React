from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

class AdaBoostClassifierModel(BaseEstimator):
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        Initialize the AdaBoostClassifierModel.

        Parameters:
        - n_estimators: The number of estimators to train (default is 50).
        - learning_rate: The weight applied to each classifier (default is 1.0).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        # Initialize AdaBoostClassifier with a default base estimator (DecisionTreeClassifier)
        self.model = AdaBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm='SAMME'  # Explicitly set to 'SAMME' to avoid deprecation
        )

    def fit(self, X, y):
        """
        Train the AdaBoostClassifier model on the given data.

        Parameters:
        - X: Training data (features).
        - y: Training labels (target).
        
        Returns:
        - self: Trained model.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X: Test data (features).

        Returns:
        - y_pred: Predicted class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Return the probability estimates for each class for the given data.

        Parameters:
        - X: Test data (features).

        Returns:
        - proba: Probability estimates for each class.
        """
        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        Return the accuracy of the model on the given data.

        Parameters:
        - X: Test data (features).
        - y: True labels (target).

        Returns:
        - score: Accuracy of the model.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Example usage of the AdaBoostClassifierModel
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Instantiate the AdaBoostClassifierModel
    model = AdaBoostClassifierModel(n_estimators=100, learning_rate=0.5)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Output predictions
    print("Predictions:", y_pred)

    # Evaluate the model's accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")
