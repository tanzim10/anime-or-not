import time
from src.model.train_model import Trainer, train, train_with_wandb
from src.data.load_data import load_image_dataset, dataloader_generator
from pathlib import Path
import torch
from torchvision import datasets
from src.data.preprocess import TransformedDataset, train_image_transformation

def main():
    # Set this flag to True to enable wandb logging
    use_wandb = True  # Change to True to enable wandb

    # Load datasets
    train_data, test_data = load_image_dataset("/Users/tanzimfarhan/Desktop/Python/Dataset")

    # Create dataloaders
    train_loader = dataloader_generator(train_data, test_data, type="train", batch_size=32)
    test_loader = dataloader_generator(train_data, test_data, type="test", batch_size=32)

    # Instantiate Trainer
    trainer = Trainer(
        num_classes=10,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        lr=0.001,
        device=None,  # Uses CUDA if available
        output_shape=10,
        model_save_path="best_model.pth"
    )

    # Track training time
    start_time = time.time()
    if use_wandb:
        results = train_with_wandb(trainer, epochs=10, use_wandb=True)
    else:
        results = train(trainer, epochs=10)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed:.2f} seconds")

    # Optionally, print results
    print("Training and validation results:")
    print(results)

if __name__ == "__main__":
    main()
