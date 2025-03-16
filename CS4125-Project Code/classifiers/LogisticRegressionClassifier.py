from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

class LogisticRegressionClassifierModel(BaseEstimator):
    def __init__(self, C=1.0, solver='liblinear'):
        # Initialize the logistic regression model with parameters
        self.C = C
        self.solver = solver
        self.model = LogisticRegression(C=self.C, solver=self.solver)

    def fit(self, X, y):
        # Train the model
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # Predict using the trained model
        return self.model.predict(X)

    def predict_proba(self, X):
        # Return probability estimates (optional)
        return self.model.predict_proba(X)

    def score(self, X, y):
        # Return the mean accuracy on the given test data and labels
        return self.model.score(X, y)
