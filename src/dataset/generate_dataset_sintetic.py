from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader
import os 
import json

def create_dummy_dataset(config_path="synthetic/synthetic_config.json"):
    loader = SyntheticDataLoader()
    
    filename = "synthetic_dataset.csv"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                filename = config.get("filename", filename)
        except Exception as e:
            print(f"Errore durante la lettura del file di configurazione: {e}")
            pass
    pure_filename = os.path.basename(filename)
    output_dir = "synthetic"
    final_path = os.path.join(output_dir, pure_filename)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Generazione dataset sintetico in corso...")
    df = loader.load()
    df.to_csv(final_path, index=False)
    print(f"[+] [SUCCESSO] Dataset creato e salvato in: '{final_path}'")

if __name__ == "__main__":
    create_dummy_dataset()