from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib  # do zapisu modelu na dysku

def train_model(X_train, y_train, X_val=None, y_val=None, save_path=None):
    """
    Train a Logistic Regression classifier.

    Args:
        X_train: matrix of input features (e.g., TF-IDF)
        y_train: labels for training
        X_val: optional, validation features
        y_val: optional, validation labels
        save_path: optional, path to save trained model

    Returns:
        model: trained classifier
    """

    # 1. Tworzymy model
    model = LogisticRegression(max_iter=1000)

    # 2. Trenujemy model na danych treningowych
    model.fit(X_train, y_train)

    # 3. Jeśli są dane walidacyjne — sprawdzamy jakość
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        print("Validation Results:\n")
        print(classification_report(y_val, y_pred))

    # 4. Jeśli podano ścieżkę — zapisujemy model do pliku
    if save_path is not None:
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")

    # 5. Zwracamy wytrenowany model
    return model
