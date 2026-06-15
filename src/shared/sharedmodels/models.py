import uuid
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class Hyperparameters(BaseModel):
    n_estimators: int
    max_depth: Optional[int] = None
    class_weight: Optional[str] = None
    max_samples: float = 1.0
    bootstrap: bool = True
    tree_type: Literal["classifier", "regressor"] = "classifier"
    target_column: Optional[str] = None

    @field_validator("max_samples")
    @classmethod
    def check_max_samples(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("max_samples deve essere compreso tra 0 (escluso) e 1 (incluso).")
        return v


class TrainingRequest(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    environment: str
    mode: str
    dataset_path: str
    dataset_type: str
    hyperparameters: Hyperparameters


class TrainingRequestWorker(BaseModel):
    url_dataset: str
    job_id: str
    task_id: str
    mode: str
    dataset_type: str
    hyperparameters: Hyperparameters
    seed: int


class InferenceRequest(BaseModel):
    """Modello Pydantic per validare le richieste di inferenza nel cluster."""
    inference_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str  # L'ID del modello addestrato persistito su DynamoDB/S3 da caricare
    data_url: str  # Il percorso (locale o S3) dei nuovi dati non etichettati
    environment: str  # 'local' o 'aws' per capire dove salvare l'output delle predizioni