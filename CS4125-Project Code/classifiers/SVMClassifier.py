from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

class SVMClassifierModel(BaseEstimator):
    def __init__(self, **params):
        # Initialize the SVC model with all parameters passed through kwargs
        self.model = SVC(**params)
    
    def fit(self, X, y):
        # Fit the model to the training data
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        # Predict on the input data
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        # Return the parameters of the SVC model
        return self.model.get_params(deep)
    
    def set_params(self, **params):
        # Set the parameters of the SVC model
        self.model.set_params(**params)
        return self
    
    def score(self, X, y):
        # Calculate the accuracy of the model on the test data
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
