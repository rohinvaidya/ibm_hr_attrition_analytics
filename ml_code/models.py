from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelFactory:
    """ Factory class for creating machine learning models. """

    @staticmethod
    def get_model(model_type):
        if model_type == 'logisitic_regression':
            return LogisticRegression(max_iter = 500)
        elif model_type == 'decision_tree':
            return DecisionTreeClassifier()
        elif model_type == 'random_forest':
            return RandomForestClassifier()
        else:
            raise ValueError(f"Unknown Model Type: {model_type}")