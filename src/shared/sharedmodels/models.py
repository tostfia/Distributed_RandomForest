import uuid
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional, Union

Environment = Literal["local", "aws"]
Mode = Literal["centralized", "federated"]
DatasetType = Literal["real", "synthetic"]


class Hyperparameters(BaseModel):
    n_estimators: int
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    class_weight: Optional[str] = None
    max_samples: float = 1.0
    bootstrap: bool = True
    tree_type: Literal["classifier", "regressor"] = "classifier"
    target_column: Optional[str] = None
    max_features: Optional[Union[str, float]] = None
    criterion: Optional[str] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    noise: Optional[float] = None
    n_informative_reg: Optional[int] = None

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

    @field_validator("min_samples_split")
    @classmethod
    def check_min_samples_split(cls, v: int) -> int:
        if v < 2:
            raise ValueError("min_samples_split deve essere un intero >= 2.")
        return v

    @field_validator("max_samples")
    @classmethod
    def check_max_samples(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("max_samples deve essere compreso tra 0 (escluso) e 1 (incluso).")
        return v

    @model_validator(mode="after")
    def clear_class_weight_for_regressor(self) -> "Hyperparameters":
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
    partition_strategy: Literal["iid", "dirichlet", "by_day"] = "iid"
    partition_alpha: Optional[float] = None
    tree_allocation_strategy: Literal["proportional", "equal"] = "proportional"


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
    dataset_type: DatasetType = "real"
    hyperparameters: Hyperparameters
    partition_strategy: Literal["iid", "dirichlet", "by_day"] = "iid"
    partition_alpha: Optional[float] = None
    tree_allocation_strategy: Literal["proportional", "equal"] = "proportional"