import os
import numpy as np
import pandas as pd
from PIL import Image

# Do this before proceeding: Download latest version
# import kagglehub
# path = kagglehub.dataset_download("rashikrahmanpritom/plant-disease-recognition-dataset")
# print("Path to dataset files:", path)

def process_image(image_path):
    """
    return a numpy array representation of the image.
    
    Args:
        image_path (str): Path to the image.
    
    Returns:
        float: The mean pixel value of the image.
    """
    image = Image.open(image_path)
    image = image.resize((128, 128)) # Resize all images to a uniform size 128 * 128
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return np.array2string(image_array) # Calculate mean pixel value


def categorize_images_to_csv(dataset_folder, output_csv):
    """
    Converts an image classification dataset organized into folders into a CSV file with:
    - ImageData (mean pixel value)
    - Label (folder name: Healthy, Powdery, Rust).
    
    Args:
        dataset_folder (str): Path to the dataset folder with subfolders for each class.
        output_csv (str): Path to save the output CSV file.
    """
    data = []

    # Iterate through subfolders (each subfolder represents a class/label)
    for label in os.listdir(dataset_folder):
        label_folder = os.path.join(dataset_folder, label)
        if os.path.isdir(label_folder):  # Only process folders
            # Process each image in the label's folder
            for filename in os.listdir(label_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(label_folder, filename)
                    image_matrix = process_image(image_path)
                    data.append({"ImageData": image_matrix, "Label": label})

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}!")


# Example usage
# image_folder = "?" #should be yourPath/train/train
# output_csv = "plant_disease_dataset.csv"   
# categorize_images_to_csv(image_folder, output_csv)
