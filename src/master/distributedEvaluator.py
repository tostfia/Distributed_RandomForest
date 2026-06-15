import pickle
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DistributedEvaluator:

    @staticmethod
    def evaluate_model(model, X_test, y_test, t_dist):

        """
        Calcola le metriche e genera il report basandosi sul modello distribuito.
        t_dist: tempo totale impiegato dal sistema distribuito per l'addestramento.
        """
        print("[*] Avvio inferenza su test set...")
        start_inferenza = time.perf_counter()
        y_pred = model.predict(X_test)
        tempo_inferenza = time.perf_counter() - start_inferenza
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics, tempo_inferenza
    
    @staticmethod
    def print_report(metrics, tempo_distribuito, tempo_inferenza):
        print("\n" + "="*75)
        print("          REPORT DI VALIDAZIONE SISTEMA DISTRIBUITO")
        print("="*75)
        print(f"  ACCURATEZZA:      {metrics['accuracy']*100:.2f}%")
        print(f"  PRECISION MEDIA:  {metrics['precision']*100:.2f}%")
        print(f"  RECALL MEDIA:     {metrics['recall']*100:.2f}%")
        print(f"  F1-SCORE:         {metrics['f1']*100:.2f}%")
        print("-" * 75)
        print(f"  TEMPO ADDESTRAMENTO DISTRIBUITO (T_dist): {tempo_distribuito:.4f}s")
        print(f"  TEMPO INFERENZA TOTALE:                 {tempo_inferenza:.4f}s")
        print("="*75)


