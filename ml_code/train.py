from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """ Train and evaluate the model. """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    # Checking if the model has the method called predict_proba
    if hasattr(model, "predict_proba"):
        # Predict proba returns a matrix of two columns (negative and positive), [:, 1] means probability of the positve class in test data
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # If model doesn't hacve predict_proba it uses decision_function for each test instance.
        y_prob = model.decision_function(X_test)

    return accuracy, cm, y_test, y_prob