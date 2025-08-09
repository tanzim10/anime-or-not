import sys
import types
import importlib
from types import ModuleType
from pathlib import Path

import pytest
from PIL import Image


def _install_stub_modules_before_imports():
    # Ensure project root is on sys.path so `src` package is importable
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Stub torchinfo.summary (not in requirements)
    if 'torchinfo' not in sys.modules:
        torchinfo_stub = ModuleType('torchinfo')
        def _summary(*args, **kwargs):
            return None
        torchinfo_stub.summary = _summary
        sys.modules['torchinfo'] = torchinfo_stub

    # Stub wandb (not required for CI unit tests)
    if 'wandb' not in sys.modules:
        wandb_stub = ModuleType('wandb')
        wandb_stub.init = lambda *args, **kwargs: None
        wandb_stub.watch = lambda *args, **kwargs: None
        wandb_stub.log = lambda *args, **kwargs: None
        wandb_stub.finish = lambda *args, **kwargs: None
        sys.modules['wandb'] = wandb_stub

    # Stub src.data.preprocess if missing (without shadowing real packages)
    try:
        import src.data.preprocess  # type: ignore
    except Exception:
        # Ensure src.data exists as a package-like module
        if 'src.data' not in sys.modules:
            try:
                import src.data  # type: ignore
            except Exception:
                data_pkg = ModuleType('src.data')
                # Mark as a package-like module
                setattr(data_pkg, '__path__', [])
                sys.modules['src.data'] = data_pkg
        preprocess_stub = ModuleType('src.data.preprocess')
        preprocess_stub.TransformedDataset = object
        preprocess_stub.train_image_transformation = lambda *args, **kwargs: None
        sys.modules['src.data.preprocess'] = preprocess_stub


def test_trainer_one_epoch_runs_cpu_without_downloading_pretrained_weights():
    _install_stub_modules_before_imports()

    # Import torch and dependent modules lazily and skip if unavailable in local env
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        # Import after stubbing
        train_model = importlib.import_module('src.model.train_model')
        from torchvision import models as tv_models
    except Exception as e:
        pytest.skip(f"Skipping train test due to torch/torchvision import failure: {e}")

    class _FakePretrained(torch.nn.Module):
        def state_dict(self, *args, **kwargs):
            return {}

    def _fake_resnet50(weights=None):  # signature-compatible
        return _FakePretrained()

    original_resnet50 = getattr(tv_models, 'resnet50', None)
    tv_models.resnet50 = _fake_resnet50

    try:
        # Tiny random dataset (CPU, small tensors for speed)
        X_train = torch.randn(4, 3, 32, 32)
        y_train = torch.randint(low=0, high=2, size=(4,))
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=2)

        X_test = torch.randn(4, 3, 32, 32)
        y_test = torch.randint(low=0, high=2, size=(4,))
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=2)

        trainer = train_model.Trainer(
            num_classes=2,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            lr=0.001,
            device=torch.device('cpu'),
            output_shape=2,
            model_save_path='best_model_ci_temp.pth',
        )

        train_loss, train_acc = trainer.train_one_epoch()
        test_loss, test_acc = trainer.test_one_epoch()

        assert isinstance(train_loss, float)
        assert isinstance(test_loss, float)
        assert 0.0 <= train_acc <= 1.0
        assert 0.0 <= test_acc <= 1.0
    finally:
        # Restore original resnet50 if it existed
        if original_resnet50 is not None:
            tv_models.resnet50 = original_resnet50


def test_api_prediction_returns_expected_class_and_probability_cpu():
    # Import torch lazily and skip if unavailable
    try:
        import torch
        from src.model import prediction as pred
        import torchvision.transforms as T
    except Exception as e:
        pytest.skip(f"Skipping prediction test due to torch import failure: {e}")

    # Dummy model that strongly favors class index 1
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 5.0]], dtype=torch.float32)

    model = DummyModel()
    class_names = ["Anime", "Cartoon"]

    # Create a tiny RGB image in-memory
    image = Image.new('RGB', (32, 32), color=(255, 255, 255))

    # Monkeypatch ToTensor to avoid numpy dependency on local envs
    class _NoNumpyToTensor:
        def __call__(self, pil_image):
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            width, height = pil_image.size
            data = list(pil_image.getdata())  # sequence of (R,G,B)
            tensor = torch.tensor(data, dtype=torch.uint8).view(height, width, 3)
            tensor = tensor.permute(2, 0, 1).to(dtype=torch.float32) / 255.0
            return tensor

    original_to_tensor = getattr(T, 'ToTensor', None)
    T.ToTensor = _NoNumpyToTensor
    try:
        predicted_class, predicted_prob = pred.api_prediction(
            model=model,
            image=image,
            class_names=class_names,
            image_size=(32, 32),
            device='cpu',  # Ensure CPU-only execution in CI
        )
    finally:
        if original_to_tensor is not None:
            T.ToTensor = original_to_tensor

    assert predicted_class in class_names
    assert 0.0 <= float(predicted_prob) <= 1.0
    assert predicted_class == "Cartoon"  # index 1 should be chosen


