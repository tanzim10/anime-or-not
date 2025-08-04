import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchinfo import summary
from src.model.models import ResNet50
import os
import tqdm
from src.data.preprocess import TransformedDataset, train_image_transformation
import wandb

class Trainer:
    def __init__(self, 
                 num_classes=10,
                 train_dataloader=None,
                 test_dataloader=None,
                 lr=0.001,
                 device=None,
                 output_shape=10,
                 model_save_path=None):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Set device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize your custom ResNet50 model
        self.model = ResNet50(num_classes=num_classes)
        self.model = self.model.to(self.device)

        # Load pretrained weights from torchvision's ResNet50
        pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Copy pretrained weights to your custom model
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_model.state_dict().items()
            if k in model_dict and k != "fc.weight" and k != "fc.bias"
        }
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # Update the final layer for our classification task
        self.model.fc = torch.nn.Linear(
            in_features=self.model.fc.in_features, out_features=output_shape, bias=True
        ).to(self.device)

        # Print model summary
        summary(
            model=self.model,
            input_size=(32, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"],
        )

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.1
        )

        # Dataloaders
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # Model save path
        self.model_save_path = model_save_path or "best_model.pth"

        # Results
        self.results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    def save_model(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.model_save_path,
        )

    def load_model(self):
        checkpoint = torch.load(self.model_save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for X, y in self.train_dataloader:
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            y_pred_class = torch.argmax(y_pred, dim=1)
            total_correct += (y_pred_class == y).sum().item()
            total_samples += len(y)
        train_loss = total_loss / len(self.train_dataloader)
        train_acc = total_correct / total_samples
        return train_loss, train_acc

    def test_one_epoch(self):
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.test_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                test_pred_logits = self.model(X)
                loss = self.criterion(test_pred_logits, y)
                test_loss += loss.item()
                test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
        test_loss = test_loss / len(self.test_dataloader)
        test_acc = test_acc / len(self.test_dataloader)
        return test_loss, test_acc

    def train_loop(self, epochs=10, save_best=True):
        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            test_loss, test_acc = self.test_one_epoch()
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
            self.results["train_loss"].append(train_loss)
            self.results["train_acc"].append(train_acc)
            self.results["test_loss"].append(test_loss)
            self.results["test_acc"].append(test_acc)
            if save_best and test_acc > best_acc:
                best_acc = test_acc
                self.save_model(epoch)
        return self.results

def train(trainer: Trainer, epochs: int = 10):
    return trainer.train_loop(epochs=epochs)

def train_with_wandb(trainer: Trainer, epochs: int = 10,use_wandb: bool =False):
    if use_wandb:
        wandb.init(
            project="AnimeOrNot",
            config = wandb.config or {
                "learning_rate": trainer.lr,
                "epochs": epochs,
                "batch_size": trainer.batch_size,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "model": "ResNet50",
            }
        )
        wandb.watch(trainer.model, log="all")
        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = trainer.train_one_epoch()
            test_loss, test_acc = trainer.test_one_epoch()
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
            trainer.results["train_loss"].append(train_loss)
            trainer.results["train_acc"].append(train_acc)
            trainer.results["test_loss"].append(test_loss)
            trainer.results["test_acc"].append(test_acc)
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                })
            if test_acc > best_acc:
                best_acc = test_acc
                trainer.save_model(epoch)
        wandb.finish()
    return trainer.results





    




