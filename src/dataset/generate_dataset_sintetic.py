import pandas as pd
import numpy as np

def create_dummy_dataset(filename="sintetic_data.csv", n_rows=10000, n_features=10):
    """
    Genera un dataset sintetico per testare il sistema distribuito.
    """
    print(f"[*] Generazione di {n_rows} righe con {n_features} feature...")
    
    # 1. Genera dati casuali (float64)
    data = np.random.rand(n_rows, n_features)
    
    # 2. Genera un target binario (0 o 1, int64)
    target = np.random.randint(0, 2, size=n_rows)
    
    # 3. Crea il DataFrame
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    df["target"] = target
    
    # 4. Salva in CSV
    df.to_csv(filename, index=False)
    print(f"[+] Dataset '{filename}' creato correttamente.")
    print(f"[+] Struttura: {df.shape}")

if __name__ == "__main__":
    create_dummy_dataset()