from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierModel(BaseEstimator):
    def __init__(self, **kwargs):
        # Initialize the model with arbitrary parameters
        self.model = RandomForestClassifier(**kwargs)
    
    def fit(self, X, y):
        # Fit the model to the training data
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        # Predict on the input data
        return self.model.predict(X)
    
    def score(self, X, y):
        # Delegate to the underlying RandomForestClassifier's score method
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        # Return the parameters of the RandomForestClassifier
        return self.model.get_params(deep)
    
    def set_params(self, **params):
        # Set the parameters of the RandomForestClassifier
        self.model.set_params(**params)
        return self
