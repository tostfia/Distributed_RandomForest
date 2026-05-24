from pydantic import BaseModel
from typing import Optional

class Hyperparameters(BaseModel):
    n_estimators: int #Numero di alberi che compongono la foresta
    max_depth: int  #Profondità massima degli alberi
    class_weight: str #Ponderazione delle classi, può essere 'balanced' per bilanciare le classi in base alla frequenza o 'balanced_subsample' per bilanciare le classi in ogni campione
    max_samples: float #Percentuale di campioni da utilizzare per addestrare ogni albero, può essere un valore compreso tra 0 e 1 o un intero che rappresenta il numero di campioni

class TrainingRequest(BaseModel):
    environment: str #Ambiente di addestramento, ad esempio 'local' o 'cloud'
    mode: str #Modalità di addestramento, ad esempio 'centralized learning' o 'federated learning'
    dataset_path: str #Percorso del dataset da utilizzare per l'addestramento
    hyperparameters: Hyperparameters #Iperparametri per l'addestramento del modello