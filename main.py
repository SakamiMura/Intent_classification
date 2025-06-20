from src.load_data import load_clinc150_data
from src.preprocessing import clean_text
from src.vectorizer import vectorize_text
from src.train import train_model
from src.visualize_results import print_detailed_summary
from sklearn.metrics import classification_report
from src.glove_vectorizer import load_glove_embeddings, vectorize_with_glove

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


    print("\n\n--- Testing GloVe Vectorizer ---")

    # 1. Załaduj GloVe embeddingi
    glove_path = "data/glove.6B.50d.txt"
    glove_embeddings = load_glove_embeddings(glove_path)

    # 2. Zamień teksty na wektory GloVe (średnia z wektorów słów)
    X_train_glove = vectorize_with_glove(cleaned_train, glove_embeddings)
    X_val_glove = vectorize_with_glove(cleaned_val, glove_embeddings)
    X_test_glove = vectorize_with_glove(cleaned_test, glove_embeddings)

    # 3. Trenuj nowy model na GloVe
    glove_model = train_model(X_train_glove, y_train, X_val_glove, y_val)

    # 4. Ewaluacja na danych testowych
    y_pred_glove = glove_model.predict(X_test_glove)

    print("\nGloVe Model - Test Predictions:")
    print(classification_report(y_test, y_pred_glove))

if __name__ == "__main__":
    main()