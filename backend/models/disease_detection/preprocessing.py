# backend/models/disease_detection/preprocessing.py
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DDPreprocessor:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size
        
        # Try to load dataset
        try:
            csv_path = Path(__file__).parent / "plant_disease_dataset.csv"
            logger.info(f"Looking for file at: {csv_path}")
            
            if csv_path.exists():
                self.df = pd.read_csv(csv_path)
                
                # Map labels
                label_mapping = {
                    "Powdery Mildew": 0,
                    "Healthy": 1,
                    "Rust": 2
                }
                
                Label = self.df["Label"].replace(label_mapping)
                self.df["label"] = Label
                logger.info("Dataset loaded successfully")
            else:
                logger.warning("Dataset file not found")
                self.df = None
        
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            self.df = None

    def process_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image_array: Input image as numpy array (RGB)
            
        Returns:
            Preprocessed image array (CHW format, normalized)
        """
        try:
            # Handle different input formats
            if len(image_array.shape) == 2:  # Grayscale
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                elif image_array.shape[2] == 3:  # RGB
                    pass
                else:
                    raise ValueError(f"Unexpected number of channels: {image_array.shape[2]}")
            else:
                raise ValueError(f"Unexpected image shape: {image_array.shape}")

            # Resize
            image_array = cv2.resize(image_array, self.target_size)
            
            # Normalize to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # Convert to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            return image_array

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Error preprocessing image: {str(e)}")