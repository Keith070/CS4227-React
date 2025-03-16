from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

class ClassifierFactory:
    def create_classifier(self, classifier_type: str):
        """ Create a classifier based on the specified type. """
        if classifier_type == "random_forest":
            return RandomForestClassifierModel()
        elif classifier_type == "adaboost":
            return AdaBoostClassifierModel()
        elif classifier_type == "svm":
            return SVMClassifierModel()
        elif classifier_type == "naive_bayes":
            return NaiveBayesClassifierModel()
        elif classifier_type == "logistic_regression":
            return LogisticRegressionModel()
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

class RandomForestClassifierModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)

class AdaBoostClassifierModel:
    def __init__(self):
        self.model = AdaBoostClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)

class SVMClassifierModel:
    def __init__(self):
        self.model = SVC()

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)

class NaiveBayesClassifierModel:
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)
