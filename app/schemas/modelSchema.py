from enum import Enum
from pydantic import BaseModel, ConfigDict

class ModelHealthResponse(BaseModel):
    status: str
    code: int

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    model_version: str
    model_size: str
    model_parameters: int

class ModelPredictionResponse(BaseModel):
    predicted_label: str
    predicted_prob: float

class DeviceType(Enum):
    """
    Enum Class to represent different tu;es of device that can be used for computation

    Attributes:
        CPU (str): Represents the CPU device type.
        CUDA (str): Represents the CUDA (GPU) device type.
        AUTO (str): Automatically selects the appropriate device type based on availability.
    """

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"

