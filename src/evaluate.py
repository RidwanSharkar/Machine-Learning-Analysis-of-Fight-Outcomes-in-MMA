from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    """Evaluate the model with accuracy score and classification report."""
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))