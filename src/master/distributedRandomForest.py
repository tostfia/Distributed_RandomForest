import numpy as np
import scipy.stats as stats

class DistributedRandomForest:
    def __init__(self):
        # La foresta globale all'inizio è solo una lista vuota
        self.trees = []

    def add_worker_trees(self, list_of_trees):
        """Metodo per unire gli alberi calcolati singolarmente dai worker."""
        self.trees.extend(list_of_trees)

    def predict(self, X_new):
        """Implementazione manuale dei Passi 5, 6 e 7 dell'algoritmo."""
        # Passo 5: Generazione delle previsioni indipendenti da ciascun albero
        all_tree_predictions = []
        for tree in self.trees:
            # Ogni albero (addestrato dal worker) fa la sua predizione locale
            pred = tree.predict(X_new)
            all_tree_predictions.append(pred)
            
        # Trasformiamo in una matrice NumPy per manipolare i dati
        # Righe = Risposte dei singoli alberi, Colonne = Record da predire
        matrix_predictions = np.array(all_tree_predictions)

        # Passo 6 & 7: Voto o Media e Output Finale
        # Se è una Classificazione: prendiamo la moda (il valore più frequente)
        final_predictions, _ = stats.mode(matrix_predictions, axis=0)
        
        return final_predictions.flatten()