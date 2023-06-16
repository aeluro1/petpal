# Some of this code was inspired by the Pytorch official tutorials: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

MODEL_PATH = "model.pth"
DATASET_PATH = "oxford-iiit-pet"

NUM_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 4
NUM_WORKERS = 4
DATASET_RATIO = 0.8 # train:validate ratio

IMG_MEAN = np.array([0.5, 0.5, 0.5])
IMG_STD = ([0.5, 0.5, 0.5])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ResNet(nn.Module):
    def __init__(self, num_classes: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet18(weights = ResNet18_Weights.DEFAULT)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        self.model = self.model.to(DEVICE)

    def forward(self, data: list) -> list:
        return self.model(data)
    
    def set_ffe(self): # fixed feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        self.model = self.model.to(DEVICE)
    
class PetDataset(Dataset):
    def __init__(self, img_path: str, labels_path: str, transform: transforms.Compose = None):
        self._img_path = img_path
        dataset_list = Path(labels_path) / "list.txt"
        if dataset_list.is_file():
            self._labels = pd.read_csv(Path(labels_path) / "list.txt", header = None)
        else:
            self._create_dataset_list()
        self._classes = self.__getclasses__()
        self._transform = transform

    def _create_dataset_list(self): # WIP
        """Get classes via traversing image directory generically (without a summary list.txt)
        """
        classes = set()
        for file in Path(self._img_path).glob("*"):
            classes.add("_".join(file.stem.split("_")[:-1]))
        classes = dict((key, val) for (key, val) in enumerate(classes))

    
    def __getclasses__(self) -> list():
        df = self._labels.drop_duplicates([1])
        df[0] = df[0].str.rsplit("_", n = 1).str.get(0)
        df = df.sort_values(df.columns[1])

        classes = dict((df[1], df[0]))

        return classes

    def __len__(self) -> int:
        """Get number of samples in dataset"""
        return len(self._labels)

    def __getitem__(self, idx: int | torch.Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = Path(self._img_path) / f"{self._labels.iloc[idx, 0]}.jpg"
        image = Image.open(img_name)
        labels = np.array([self._labels.iloc[idx, 1]]) # Use breeds as classifiers
        sample = {"image": image, "class": labels}

        if self._transform:
            sample["image"] = self._transform(sample["image"])

        return sample
    
    @property
    def transform(self):
        return self._transform

    
def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int = -1):
    print(f"[Epoch {epoch}] Saving...")

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }

    torch.save(checkpoint, MODEL_PATH)

def load_model(model: nn.Module, optimizer) -> nn.Module:
    if not Path(MODEL_PATH).is_file():
        return (model, optimizer)

    checkpoint = torch.load(MODEL_PATH)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return (model, optimizer)

def train_model(model: nn.Module,
                dataloaders: dict[str, DataLoader],
                criterion: torch.nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler):
    best_acc = 0.0
    start_time = time()

    for epoch in (t := tqdm(range(NUM_EPOCHS))):

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for (inputs, labels) in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(predicted == labels).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_correct / len(dataloaders[phase].dataset)

            t.set_description(f"[Epoch: {epoch + 1}/{NUM_EPOCHS}][{phase}] Loss: {epoch_loss:.3f} | Accuracy: {epoch_acc:.3f}")

            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_PATH)
        
    total_time = time() - start_time
    print(f"Total training time: {total_time // 60}m {total_time % 60}s\n")
    print(f"Best accuracy: {best_acc:.3f}")

    load_model(model, optimizer)

    return model

def test_model(model: nn.Module, data: DataLoader) -> float:

    accuracy = []

    model.eval()
    with torch.no_grad():
        for (input, result_actual) in data:
            input = input.to(DEVICE)
            result_act = result_actual.to(DEVICE)
            result_pred = model(input)
            result_pred = torch.argmax(result_pred, -1)
            accuracy.append(((result_pred == result_act).nonzero()).size(0)/result_act.size(0))
    model.train()

    return sum(accuracy) / len(accuracy)

def show_model(model: nn.Module, datasets: DataLoader, img_num: int = 6):
    training_state = model.training
    img_count = 0
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for (i, (inputs, labels)) in enumerate(datasets["test"]):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                img_count += 1
                ax = plt.subplot(img_num // 3, 3, img_count)
                ax.axis("off")
                ax.set_title(f"Predicted: {datasets.dataset._classes[predicted[j]]}")
                imshow(inputs.cpu().data[j])

                if img_count == img_num:
                    model.train(mode = training_state)
                    return
        model.train(mode = training_state)

def predict_image(model: nn.Module, datasets, path: str):
    training_state = model.training
    model.eval()

    img = Image.open(path)
    # img = img_transforms["test"](img)
    img = img.unsqueeze(0)
    img = img.to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        ax.set_title(f"Predicted: {datasets.dataset._classes[predicted[0]]}")
        imshow(img.cpu().data[0])
        model.train(mode = training_state)

def imshow(img: torch.Tensor):
    img = img.numpy().transpose((1, 2, 0))
    img = IMG_STD * img + IMG_MEAN
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    # plt.show()
    
def clean_dataset(img_transform: transforms.Compose):
    for file in (Path(DATASET_PATH) / "images").glob("*"):
        try:
            img_transform(Image.open(file))
        except:
            file.unlink()

def setup_transforms() -> dict[transforms.Compose]:
    img_transforms = dict()
    img_transforms["train"] = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean = IMG_MEAN, std = IMG_STD)])
    img_transforms["test"] = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean = IMG_MEAN, std = IMG_STD)])
    return img_transforms

def main(args: tuple):
    img_transform = setup_transforms()
    if args.clean:
        clean_dataset(img_transform)

    if not args.train and not args.validate:
        return
    
    img_path = Path(DATASET_PATH) / "images"
    labels_path = Path(DATASET_PATH) / "annotations"
    dataset = PetDataset(img_path, labels_path, img_transform)

    train_size = round(DATASET_RATIO * len(dataset))
    test_size = len(dataset) - train_size

    dataset_train, dataset_test = random_split(dataset, (train_size, test_size))
    datasets = dict()
    datasets["train"] = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
    datasets["test"] = DataLoader(dataset_test, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS)

    model = ResNet().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9) # For fine-tuned version
    # optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.fc.parameters(), lr = LEARNING_RATE, momentum = 0.9) # For fine-tuned version
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
    load_model(model, optimizer)

    if args.train:
        train_model(model, datasets, criterion, optimizer, scheduler)

    if args.validate:
        accuracy = test_model()
        print(f"Accuracy of model: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ResNet-Pet")
    parser.add_argument(
        "--train", "-t",
        action = "store_true",
        help = "to train model",
    )
    # parser.add_argument(
    #     "--validate", "-v",
    #     action = "store_true",
    #     help = "to validate model",
    # )
    parser.add_argument(
        "--clean", "-c",
        action = "store_true",
        help = "to clean dataset",
    )
    args = parser.parse_args()

    main(args)