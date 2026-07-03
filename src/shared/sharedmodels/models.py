import uuid
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional

Environment = Literal["local", "aws"]
Mode = Literal["centralized", "federated"]
DatasetType = Literal["real", "synthetic"]


class Hyperparameters(BaseModel):
    n_estimators: int
    max_depth: Optional[int] = None
    class_weight: Optional[str] = None
    max_samples: float = 1.0
    bootstrap: bool = True
    tree_type: Literal["classifier", "regressor"] = "classifier"
    target_column: Optional[str] = None

    @field_validator("n_estimators")
    @classmethod
    def check_n_estimators(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_estimators deve essere un intero positivo.")
        return v

    @field_validator("max_depth")
    @classmethod
    def check_max_depth(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("max_depth, se specificato, deve essere un intero positivo.")
        return v

    @field_validator("max_samples")
    @classmethod
    def check_max_samples(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("max_samples deve essere compreso tra 0 (escluso) e 1 (incluso).")
        return v

    @model_validator(mode="after")
    def clear_class_weight_for_regressor(self) -> "Hyperparameters":
        # class_weight ha senso solo per la classificazione: lo azzeriamo qui
        # cosi' la regola vale per qualsiasi punto del codice crei l'oggetto,
        # non solo per il ramo di main.py che se ne ricorda di farlo.
        if self.tree_type == "regressor":
            self.class_weight = None
        return self


class TrainingRequest(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    environment: Environment
    mode: Mode
    dataset_path: str
    dataset_type: DatasetType
    hyperparameters: Hyperparameters
    seed: int = 123


class TrainingRequestWorker(BaseModel):
    url_dataset: str
    job_id: str
    task_id: str
    mode: Mode
    dataset_type: DatasetType
    hyperparameters: Hyperparameters
    seed: int = 123


class InferenceRequest(BaseModel):
    inference_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_type: Literal["INFERENCE"] = "INFERENCE"
    job_id: str
    data_url: Optional[str] = None
    environment: Environment
    hyperparameters: Hyperparameters