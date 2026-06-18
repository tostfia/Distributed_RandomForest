from src.shared.utilities.loader.synthetic_dataloader import SyntheticDataLoader

def create_dummy_dataset(filename="sintetic_data.csv", n_rows=10000):
    loader = SyntheticDataLoader(n_samples=n_rows, target_column="Label")
    df = loader.load()
    df.to_csv(filename, index=False)
    print(f"[+] Dataset '{filename}' creato usando SyntheticDataLoader.")

if __name__ == "__main__":
    create_dummy_dataset()