import io
import torch
from fastapi import File, HTTPException, UploadFile
from PIL import Image
from torchvision import models

from app.schemas.modelSchema import ModelHealthResponse, ModelInfoResponse, ModelPredictionResponse
from src.model.prediction import predict_and_plot
from src.model.models import ResNet50
from app.exceptions.custom_exceptions import InvalidImageFileException
from src.model.prediction import api_prediction

class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = ResNet50(num_classes=2)
        checkpoint = torch.load('best_model.pth', map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        # Remove final layer weights
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        model.load_state_dict(state_dict, strict=False)

        self.model = model
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.model_size = sum(p.element_size() * p.numel() for p in self.model.parameters()) / (1024**2)  # in MB
    
    def check_api_status(self) -> ModelHealthResponse:
        try:
            _ = self.model(2)
            return ModelHealthResponse(status="API is ready for serving model predictions", code =200)
        except Exception as e:
            raise HTTPException(status_code=503, detail= "Service Unavailable: " + str(e))
    
    def get_model_info(self) -> ModelInfoResponse:
        model_info = {
            "model_name": "resNet50",
            "model_version": "1.0",
            "model_size": f"{self.model_size:.2f} MB",
            "model_parameters": self.total_params,
        }
        return model_info
    
    async def make_prediction(self, file:UploadFile = File(...)):
        if file.content_type is None or not file.content_type.startswith("image/"):
            raise InvalidImageFileException
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        class_names = ["Anime", "Cartoon"]
        pred_class, pred_prob = api_prediction(self.model, img, class_names, device=str(self.device))
        # Ensure pred_prob is a float
        pred_prob_float = float(pred_prob)
        return ModelPredictionResponse(predicted_label=pred_class, predicted_prob=pred_prob_float)

