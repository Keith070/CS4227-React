# strategies/AdaBoostStrategy.py
from sklearn.ensemble import AdaBoostClassifier
from strategies.ClassificationStrategy import ClassificationStrategy
from sklearn.metrics import accuracy_score

class AdaBoostStrategy(ClassificationStrategy):
    def __init__(self):
        self.model = AdaBoostClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
