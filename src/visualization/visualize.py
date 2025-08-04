import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np
from pathlib import Path


def visualize_image(image_path: str):
    """
    Visualize data from the specified image path.

    Args:
        image_path (str): Path to the directory containing image files.
    """

    image_path_list = list(image_path.glob("**/*.jpg"))  # Assuming images are in JPG format
    if not image_path_list:
        raise FileNotFoundError(f"No images found in {image_path}")
    else:
        random_image_path = random.choice(image_path_list)
        image_class = random_image_path.parent.parent.name
        image = Image.open(random_image_path)

        # Turn the image into a numpy array
        image_array = np.asarray(image)

        # Visualize the image
        plt.figure(figsize=(10, 7))
        plt.imshow(image_array)
        plt.title(f"Image class: {image_class} | Image shape: {image_array.shape} -> [H, W, C]")
        plt.axis(False)
        plt.show()

def plot_transformed_image(image_paths, transform, n=3, seed=150):
    """
    Plot a series of random images and then it will open n image paths
    and plot them with the same transform

    Args:
        image_paths (list): List of image paths to plot
        transform (callable): Transform to apply to the images
        n (int): Number of images to plot
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, n)
    plt.figure(figsize=(15, 10))
    return None


def plot_loss_curves(results):
    """
    Plot loss curves for training and validation
    """
    # Get the loss values from the results dictionary
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    # Get the accuracy values from the results dictionary
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Plot the loss curves
    plt.plot(loss, label="Train loss")
    plt.plot(test_loss, label="Test loss")
    plt.title("Loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot the accuracy curves
    plt.plot(accuracy, label="Train accuracy")
    plt.plot(test_accuracy, label="Test accuracy")
    plt.title("Accuracy curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()


if __name__ == "__main__":

    # Example usage
    data_path = Path("/Users/tanzimfarhan/Desktop/Python/Dataset/AONTrainingData")
    if not data_path.is_dir():
        raise FileNotFoundError(f"Directory {data_path} does not exist.")
    
    visualize_image(data_path)



    

