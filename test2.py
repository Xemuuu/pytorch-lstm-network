import pandas as pd

def analyze_stock_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return

    required_columns = {'date', 'symbol', 'open', 'close', 'low', 'high', 'volume'}
    if not required_columns.issubset(df.columns):
        print(f"Brakuje wymaganych kolumn: {required_columns - set(df.columns)}")
        return


    mean_close = df['close'].mean()
    mean_volume = df['volume'].mean()
    std_close = df['close'].std()


    print(f"Średnia cena zamknięcia (close): {mean_close:.2f}")
    print(f"Średni wolumen (volume): {mean_volume:.2f}")
    print(f"Odchylenie standardowe ceny zamknięcia (close): {std_close:.2f}")


if __name__ == "__main__":
    analyze_stock_data("data/NVDA.csv")
