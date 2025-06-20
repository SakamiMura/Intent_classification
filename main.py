from src.load_data import load_clinc150_data
from src.preprocessing import clean_text
from src.vectorizer import vectorize_text
from src.train import train_model
from src.visualize_results import print_detailed_summary
from sklearn.metrics import classification_report

def main():
    # 1. Wczytanie danych z pliku JSON
    train_df, val_df, test_df, oos_df = load_clinc150_data("data/data_full.json")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # 2. Czyszczenie tekstów
    cleaned_train = [clean_text(text) for text in train_df['text']]
    cleaned_val = [clean_text(text) for text in val_df['text']]
    cleaned_test = [clean_text(text) for text in test_df['text']]

    # 3. Wektoryzacja tekstów przy użyciu TF-IDF
    X_train, X_val, _, vectorizer = vectorize_text(cleaned_train, cleaned_val)

    # 4. Wyciągnięcie etykiet (targetów)
    y_train = train_df['intent']
    y_val = val_df['intent']
    y_test = test_df['intent']

    # 5. Trenowanie modelu
    model = train_model(X_train, y_train, X_val, y_val)

    # 6. Testowanie modelu na danych testowych
    X_test = vectorizer.transform(cleaned_test)
    y_pred = model.predict(X_test)
    
    print("Test Predictions:")
    print(classification_report(y_test, y_pred))

    # 7. Enhanced summary analysis (text only)
    y_val_pred = model.predict(X_val)
    print_detailed_summary(y_val, y_val_pred, "Validation")
    print_detailed_summary(y_test, y_pred, "Test")

    # (opcjonalnie) zapisanie modelu:
    # train_model(..., save_path="models/intent_model.pkl")

if __name__ == "__main__":
    main()