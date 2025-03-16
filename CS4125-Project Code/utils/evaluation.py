from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test, subject):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)

    if len(set(y_test)) == 2:  # Binary classification
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"ROC-AUC Score: {auc:.2f}")

    subject.notify({
        "accuracy": f"{accuracy * 100:.2f}%",
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report
    })
