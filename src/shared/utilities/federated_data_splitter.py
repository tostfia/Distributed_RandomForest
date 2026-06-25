import os
import numpy as np
import pandas as pd

class FederatedDataSplitter:
    def __init__(self, target_column="Label", test_size=0.20, random_state=123):
        self.target_column = target_column
        # Usiamo il tuo splitter stratificato esistente
        from src.shared.utilities.datasplitter import StratifiedDataSplitter
        self.central_splitter = StratifiedDataSplitter(target_column=target_column, test_size=test_size, random_state=random_state)

    def split_and_shard(self, loader, num_workers):
        print(f"\n[Splitter] Distribuzione shard su {num_workers} nodi...")
        df = loader.load()
        train_df, test_df = self.central_splitter.split(df)
        
        train_shards = np.array_split(train_df, num_workers)
        test_shards = np.array_split(test_df, num_workers)

        for i in range(num_workers):
            # Ogni worker ha la sua cartella isolata
            cache_dir = f"./Worker-Locale-0{i+1}_cache"
            os.makedirs(cache_dir, exist_ok=True)
            train_shards[i].to_csv(os.path.join(cache_dir, "train_shard.csv"), index=False)
            test_shards[i].to_csv(os.path.join(cache_dir, "test_shard.csv"), index=False)
            print(f"[Splitter] Scritto shard per Worker-Locale-0{i+1}")