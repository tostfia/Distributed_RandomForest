"""
Stampa il numero di righe di ciascun CSV sorgente, per decidere un valore
sensato di target_rows_per_day prima di attivare il campionamento
ribilanciato in run_baseline.py. Non modifica né campiona nulla: usa solo
DatasetDAO.count_rows() (S3 Select lato server per S3, conteggio righe per
locale), quindi è economico da eseguire anche più volte.

Uso:
    python -m src.baseline.check_row_counts
"""
import os
from src.shared.utilities.loader.raw_csvdataloader import RawCSVDataLoader

def main():
    data_folder = os.environ.get("DATASET_LOCAL_PATH", "./dataset_cache")
    loader = RawCSVDataLoader(data_url=data_folder, sample_fraction=1.0, dataset_seed=123)
    sources = loader._discover_sources()

    print(f"Conteggio righe per {len(sources)} sorgenti in '{data_folder}':\n")
    counts = loader._get_row_counts(sources) if hasattr(loader, "_get_row_counts") else None
    if counts is None:
        # Fallback se preferisci non usare il metodo "privato" direttamente
        counts = {}
        for s in sources:
            dao = loader._s3_dao if loader._is_s3_path(s) else loader._local_dao
            counts[s] = dao.count_rows(s)

    for source, n in sorted(counts.items()):
        print(f"  {os.path.basename(source):<55} {n:>10,} righe".replace(",", "."))

    total = sum(counts.values())
    print(f"\n  Totale: {total:,} righe su {len(sources)} file".replace(",", "."))
    print(f"  Media per file: {total // len(sources):,} righe".replace(",", "."))
    print(f"  Minimo: {min(counts.values()):,} righe".replace(",", "."))
    print(f"  Massimo: {max(counts.values()):,} righe".replace(",", "."))
    print("\nScegli target_rows_per_day guardando soprattutto il MINIMO: se lo superi, "
          "quel file contribuirà con TUTTE le sue righe (fraction=1.0), non con la quota "
          "target — è normale, ma tienilo presente nel bilanciamento finale.")

if __name__ == "__main__":
    main()