from src.load_data import load_clinc150_data
from src.preprocessing import clean_text

def preview_cleaning():
    # Wczytujemy dane
    train_df, val_df, test_df, oos_df = load_clinc150_data("data/data_full.json")

    print(f"Training set size: {len(train_df)} samples\n")

    # Podglądamy pierwsze 5 tekstów
    for i in range(5):
        original = train_df.iloc[i]['text']
        cleaned = clean_text(original)
        print(f"--- Example {i + 1} ---")
        print("Original:", original)
        print("Cleaned: ", cleaned)
        print()

if __name__ == "__main__":
    preview_cleaning()
