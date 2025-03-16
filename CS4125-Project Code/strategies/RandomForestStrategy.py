from sklearn.ensemble import RandomForestClassifier
from strategies.ClassificationStrategy import ClassificationStrategy

class RandomForestStrategy(ClassificationStrategy):
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_test, y_pred)
