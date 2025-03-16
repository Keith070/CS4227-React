from sklearn.model_selection import GridSearchCV
from classifiers.RandomForestClassifier import RandomForestClassifierModel
from classifiers.SVMClassifier import SVMClassifierModel
from classifiers.AdaBoostClassifier import AdaBoostClassifierModel
from classifiers.NaiveBayesClassifier import NaiveBayesClassifierModel
from classifiers.LogisticRegressionClassifier import LogisticRegressionModel

def choose_classifier_with_tuning(classifier_choice, X_train, y_train):
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
        model = NaiveBayesClassifierModel()
        param_grid = {'alpha': [0.1, 0.5, 1.0]}
    elif classifier_choice == 'logistic_regression':
        model = LogisticRegressionModel()
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    else:
        raise ValueError("Invalid classifier choice")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
