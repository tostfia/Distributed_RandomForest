from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
import os 
import json
def create_dummy_dataset(config_path = "synthetic/synthetic_config.json"):
    loader = SyntheticDataLoader()
    output_dir = "synthetic"
    filename = "synthetic_dataset.csv"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                filename = config.get("filename", filename)
        except Exception as e:
            print(f"Errore durante la lettura del file di configurazione: {e}")
            pass
    final_path = os.path.join(output_dir, os.path.basename(filename))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Creazione dataset sintetico '{filename}'...")
    df = loader.load()
    df.to_csv(filename, index=False)
    print(f"[+] Dataset '{filename}' [SUCCESSO] Dataset creato.")

if __name__ == "__main__":
    create_dummy_dataset()