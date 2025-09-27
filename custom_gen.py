import os
import numpy as np
from typing import List, Generator, Tuple, Any
import random # Import for shuffling the indices
import tensorflow as tf # Added for one-hot encoding utility

# Define the number of classes globally for use in the OHE step
N_CLASSES = 4 

# --- Helper Function ---

def load_img(img_dir: str, img_list: List[str]) -> np.ndarray:
    """
    Loads images/masks from a specified directory, assuming they are stored as 
    NumPy arrays (.npy files).
    
    CRITICAL FIX: Converts loaded data to float32, ensures a channel dimension exists 
    for 3D volumetric data, applies normalization for input images, and ensures 
    the correct number of channels (3 for input images, based on your model).

    Args:
        img_dir (str): The directory path where the .npy files are located.
        img_list (List[str]): A list of image filenames to load.
        
    Returns:
        np.ndarray: A NumPy array containing the stacked images/masks.
    """
    images = []
    
    # Check if this directory is for images (X) or masks (Y).
    is_input_image_dir = "image" in img_dir.lower() 
    
    for image_name in img_list:
        if image_name.lower().endswith('.npy'):
            try:
                image_path = os.path.join(img_dir, image_name)
                
                # Load and enforce float32
                image = np.load(image_path).astype(np.float32)
                
                # --- APPLY DATA NORMALIZATION (Input Images only) ---
                if is_input_image_dir:
                    # Input (X) Normalization: Scale to [0, 1] if values are high.
                    if image.max() > 1.0 and image.size > 0:
                       image = image / image.max()
                       
                # --- CHANNEL DIMENSION CHECK AND FIX ---
                # Check 1: If the volume is 3D (D, H, W), expand its last axis to (D, H, W, 1).
                # This ensures every volume has a channel axis, regardless of content.
                if image.ndim == 3:
                    image = np.expand_dims(image, axis=-1)

                # Check 2 (BraTS Specific): If loading images (X) and there are 4 channels 
                # (which is standard for BraTS raw data), slice it to 3 channels to match 
                # your model's IMG_CHANNELS=3 setting (channels 0, 1, and 2).
                if is_input_image_dir and image.ndim == 4 and image.shape[-1] > 3:
                    # Keep channels 0, 1, 2 (flair, t1ce, t2 - based on your main script's visualization)
                    image = image[..., :3]

                # Check 3 (Segmentation Mask OHE): If loading masks (Y), we must ensure 
                # they are one-hot encoded if they are not already.
                # If the mask is 4D but has only 1 channel, it means it is label-encoded 
                # and needs OHE to match the model's 4 output classes.
                if not is_input_image_dir and image.ndim == 4 and image.shape[-1] == 1:
                    # Remove the single channel dimension (D, H, W, 1) -> (D, H, W)
                    image = np.squeeze(image, axis=-1)
                    # Convert to one-hot encoding (D, H, W) -> (D, H, W, N_CLASSES)
                    # Ensure indices are integers before OHE
                    image = tf.keras.utils.to_categorical(
                        image.astype(np.int32), num_classes=N_CLASSES
                    ).astype(np.float32)
                
                images.append(image)
            except Exception as e:
                # Print error and skip file if loading fails
                print(f"Error loading {image_name} at {image_path}: {e}. Skipping file.")
                continue 
                
    # Stack all loaded images/masks into a single NumPy array (creating the batch)
    if not images:
        return np.array([]) 
        
    images = np.array(images)
    return images

# --- Keras Data Generator ---

def imageLoader(
    img_dir: str, 
    img_list: List[str], 
    mask_dir: str, 
    mask_list: List[str], 
    batch_size: int
) -> Generator[Tuple[np.ndarray, np.ndarray], Any, None]:
    """
    A Keras-compatible data generator that yields batches of image (X) and mask (Y) data.
    
    This generator runs indefinitely, looping through the dataset for epoch-based training.
    Shuffling is performed at the start of each epoch.

    Args:
        img_dir (str): Directory containing the input images (X).
        img_list (List[str]): List of input image filenames.
        mask_dir (str): Directory containing the target masks (Y).
        mask_list (List[str]): List of target mask filenames (must correspond to img_list).
        batch_size (int): The number of samples to yield in each batch.
        
    Yields:
        Tuple[np.ndarray, np.ndarray]: A tuple (X_batch, Y_batch) for Keras model.fit().
    """
    L = len(img_list)

    # Sanity Check: Ensure the lists are aligned
    if L != len(mask_list):
        raise ValueError("Image list and mask list must have the same length.")

    # Keras requires the generator to be infinite (while True)
    while True:
        # 1. Shuffle the data indices at the start of a new epoch
        indices = np.arange(L)
        np.random.shuffle(indices)

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            # Determine the batch slice limit using the total length L
            limit = min(batch_end, L)
            
            # Get the indices for the current batch from the shuffled array
            batch_indices = indices[batch_start:limit]
            
            # Retrieve the corresponding filenames using the shuffled indices
            X_filenames = [img_list[i] for i in batch_indices]
            Y_filenames = [mask_list[i] for i in batch_indices]
                            
            # Load the batch of input images (X)
            X = load_img(img_dir, X_filenames)
            
            # Load the corresponding batch of masks (Y)
            Y = load_img(mask_dir, Y_filenames)
            
            # Check if loading was successful and the batch is non-empty
            if X.size > 0 and Y.size > 0 and len(X) == len(Y):
                # Yield the batch (a tuple of two numpy arrays: (inputs, targets))
                yield (X, Y) 
            
            # Move to the next batch
            batch_start += batch_size    
            batch_end += batch_size
