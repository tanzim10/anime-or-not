from typing import List, Dict, Any, Tuple, Optional
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image

class_names= ["Anime", "Cartoon"]

#Load the trained model
def load_model(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 
                               out_features=len(class_names), 
                               bias=True).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

#Take in a trained model, class names, and image path, a transform, and a device
def predict_and_plot(model: torch.nn.Module,
                     image_path:str,
                     class_names: List[str],
                     image_size: Tuple[int,int],
                     transform: Optional[transforms.Compose] = None,
                     plot: bool = False,
                     device: str = "cuda"):
    
    #Load the image
    image = Image.open(image_path)
    # Create transform if not provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    # Make sure the model is on the target device
    model.to(device)
    # Put the model in eval mode
    model.eval()
    
    #Turn on model evaluation mode
    with torch.inference_mode():
        # Transform the image and add an extra dimension to image
        transformed_image = transform(image).unsqueeze(0).to(device)
        
        # Make a prediction on the image with extra dimension and send it to the target device
        model_pred_logits = model(transformed_image.to(device))

    #Convert the logits to prediction probabilities
    model_pred_probs = torch.softmax(model_pred_logits, dim=1)

    #Get the predicted class
    model_pred_label = torch.argmax(model_pred_probs, dim=1)

    #Get the predicted class name
    model_pred_class = class_names[model_pred_label]

    #Plot the image and prediction
    if plot:
        plt.figure(figsize=(10, 7))
        plt.imshow(image)
        plt.title(f"Predicted: {model_pred_class}")
        plt.axis(False)
        plt.show()

    return model_pred_class, model_pred_probs


def api_prediction(model: torch.nn.Module, image: Image.Image, class_names: list, image_size: tuple = (224, 224), device: str = "cuda"):
    """
    Predict the class name and probability from a PIL Image using the provided model.
    Args:
        model: Trained torch model.
        image: PIL Image object (already in RGB mode).
        class_names: List of class names.
        image_size: Target image size for preprocessing.
        device: Device to run prediction on.
    Returns:
        Tuple (predicted_class_name, predicted_probability)
    """

    # Define transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)

    # Move model and tensor to device
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Set model to eval mode
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        pred_prob = probs[0, pred_idx].item()

    return pred_class, pred_prob



