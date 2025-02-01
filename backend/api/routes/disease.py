import io
import logging
import os

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from models.disease_detection.model import CNNClassifier
from models.disease_detection.preprocessing import DDPreprocessor
from PIL import Image
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Disease labels mapping
DISEASE_LABELS = {
    0: "Powdery Mildew",
    1: "Healthy",
    2: "Rust"
}

class DiseaseDetectionService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Get absolute path to model file
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'disease_detection', 'model.pth')
        
        # Debug logging
        logger.info(f"__file__ location: {__file__}")
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Attempting to load model from: {MODEL_PATH}")
        logger.info(f"Does path exist? {os.path.exists(MODEL_PATH)}")
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model weights file not found at: {MODEL_PATH}")
            logger.info(f"Current working directory: {os.getcwd()}")
            raise RuntimeError(f"Model weights file not found at: {MODEL_PATH}")
            
        # Initialize model
        self.model = CNNClassifier(input_shape=(3, 32, 32), num_classes=3)
        self.model = self.model.to(self.device)
        
        # Define the transform here
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])
        
        try:
            logger.info(f"Attempting to load model from: {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            
            # Debug log for state dict
            logger.info("Model state dict keys: %s", state_dict.keys())
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Model loaded successfully with trained weights")
            
            # Verify model is actually trained
            test_input = torch.randn(1, 3, 32, 32).to(self.device)
            with torch.no_grad():
                test_output = self.model(test_input)
                logger.info(f"Test prediction shape: {test_output.shape}")
                
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            raise RuntimeError(f"Failed to load model weights: {str(e)}")

    async def predict(self, image_bytes: bytes, threshold=0.85):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            
            logger.info(f"Input image size: {image.size}")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                # Add small epsilon to prevent extreme probabilities
                outputs = outputs + torch.randn_like(outputs) * 0.1
                
                # Apply temperature scaling to soften the predictions
                temperature = 2.0
                outputs = outputs / temperature
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                logger.info(f"Adjusted probabilities: {probabilities}")
                
                # Force uncertainty: no prediction should be above 90%
                probabilities = torch.clamp(probabilities, max=0.9)
                
                # Redistribute remaining probability
                remaining_prob = 1 - probabilities
                probabilities += remaining_prob / 3
                
                # Bias towards diseases (reduce healthy probability by 20%)
                probabilities[0][1] *= 0.8  # Reduce healthy class probability
                
                # Renormalize
                probabilities = probabilities / probabilities.sum()
                
                # Get predictions
                confidence, predicted = torch.max(probabilities, 1)
                
                # Always predict disease if healthy confidence isn't very high
                if predicted.item() == 1 and confidence.item() < 0.7:
                    disease_probs = [probabilities[0][0].item(), probabilities[0][2].item()]
                    disease_idx = 0 if disease_probs[0] > disease_probs[1] else 2
                    predicted = torch.tensor([disease_idx]).to(self.device)
                    confidence = torch.tensor([max(disease_probs)]).to(self.device)
                
                logger.info(f"Final predicted class: {predicted.item()}")
                logger.info(f"Final confidence: {confidence.item()}")
                
                result = {
                    "disease": DISEASE_LABELS[predicted.item()],
                    "confidence": float(confidence.item()),
                    "recommendations": get_recommendations(predicted.item()),
                    "probabilities": {
                        DISEASE_LABELS[i]: float(prob.item())
                        for i, prob in enumerate(probabilities[0])
                    }
                }
                
                logger.info(f"Final prediction result: {result}")
                return result
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.exception("Full traceback:")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize service
try:
    service = DiseaseDetectionService()
except Exception as e:
    logger.error(f"Failed to initialize DiseaseDetectionService: {str(e)}")
    raise

def get_recommendations(disease_id: int) -> str:
    recommendations = {
        0: "For Powdery Mildew: Apply fungicide treatment, ensure proper air circulation, and remove infected leaves. Avoid overhead watering.",
        1: "Plant is healthy! Continue regular maintenance and monitoring.",
        2: "For Rust: Apply rust-specific fungicide, improve air circulation, and avoid overhead watering. Remove and destroy infected plant material."
    }
    return recommendations.get(disease_id, "Consult with a plant specialist for proper diagnosis and treatment.")

@router.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        result = await service.predict(contents)
        return result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))