import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
from torchvision import models
from multiprocessing import freeze_support
from sklearn.utils import shuffle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchinfo
import requests as reqs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = open(r"food-101/meta/classes.txt", 'r').read().splitlines()
classes_21 = classes[:20]

class Label_encoder:
    def __init__(self, labels):
        self.labels = {label: idx for idx, label in enumerate(labels)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels.get(label)

encoder_21 = Label_encoder(classes_21)
encoder_21.get_label(0), encoder_21.get_idx( encoder_21.get_label(0) )

def prep_df(path: str) -> pd.DataFrame:
    array = open(path, 'r').read().splitlines()

    # Getting the full path for the images
    img_path = r"food-101/images/"
    full_path = [img_path + img + ".jpg" for img in array]

    # Splitting the image index from the label
    imgs = []
    for img in array:
        img = img.split('/')

        imgs.append(img)

    imgs = np.array(imgs)

    for idx, img in enumerate(imgs):
        if encoder_21.get_idx(img[0]) is None:
            imgs[idx, 0] = "other"
    
    # Converting the array to a data frame
    imgs = pd.DataFrame(imgs[:, 0], imgs[:,1], columns=['label'])
            
    # Adding the full path to the data frame
    imgs['path'] = full_path

    # Randomly shuffling the order of the data in the dataframe
    imgs = shuffle(imgs)

    return imgs

train_imgs = prep_df(r'food-101/meta/train.txt')
test_imgs = prep_df(r'food-101/meta/test.txt')

# Data augmentation for training
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# Data augmentation for testing
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Creation of a DataLoader for the Dataset
class Food21(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_name = self.dataframe.path.iloc[idx]
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        label = encoder_21.get_idx(self.dataframe.label.iloc[idx])

        if self.transform:
            image = self.transform(image)

        return image, label
    
train_dataset = Food21(train_imgs, transform=train_transforms)
test_dataset = Food21(test_imgs, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=150, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False, num_workers=8)

# Creating the ML Model
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

        self.fc = nn.Linear(101, 21)
    
    def forward(self, x):
        return self.fc(x)

weights = models.DenseNet201_Weights.IMAGENET1K_V1
head = Head()

model = models.densenet201(weights = weights)

if __name__ == "__main__":
    freeze_support()
    for param in model.parameters():
        param.requires_grad = False

    checkpoint_path = r"./food_classifier.pt"
    classifier = nn.Sequential(
    nn.Linear(1920,1024),
    nn.LeakyReLU(),
    nn.Linear(1024,101),
    )
    model.classifier = classifier
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=False)
    model = nn.DataParallel(nn.Sequential(model, head))
    torchinfo.summary(model, (1, 3, 224, 224))
    
    num_epochs = 35
    lr = 1e-3

    # loss
    loss_fn = nn.CrossEntropyLoss()

    # all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999])
    
    def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
        model.train()
        train_loss, train_acc = 0, 0
        print("--> Training Progress")
        # Loop through data loader data batches
        for batch, (X, y) in enumerate(tqdm(dataloader)):
             images, labels = X.to(device), y.to(device)
             # 1. Forward pass
             y_pred = model(images)

      # 2. Calculate  and accumulate loss
             loss = loss_fn(y_pred, labels)
             train_loss += loss.item()

      # 3. Optimizer zero grad
             optimizer.zero_grad()

      # 4. Loss backward
             loss.backward()

      # 5. Optimizer step
             optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
             y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
             train_acc += (y_pred_class == labels).sum().item()/len(y_pred)
        
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    
    def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
        model.eval()
        test_loss, test_acc = 0, 0 
        with torch.inference_mode():
            print("--> Testing Progress")
      # Loop through DataLoader batches
            for batch, (X, y) in enumerate(tqdm(dataloader)):
          # Send data to target device
                images, labels = X.to(device), y.to(device)

          # 1. Forward pass
                test_pred_logits = model(images)

          # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, labels)
                test_loss += loss.item()

          # Calculate and accumulate accuracy
                test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)

                test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))
                
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc
    
    def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          history: dict()):
        # Create empty results dictionairy if the provided dictionairy is empty
        if history == None:
            history = {
          "train_loss": [],
          "train_acc": [],
          "test_loss": [],
          "test_acc": [],
          'best train acc': (0, 0),
          "best_model": dict()
          }
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
            test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
            print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
          f"\n\n=============================\n"
      )

      # Update results dictionary
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            if test_acc > max(history["test_acc"]):
                history["best_model"] = model.state_dict()

            if test_acc > 0.96:
                break
            
        return model, history
    
    model, history = train(model, train_loader, test_loader, optimizer, loss_fn, num_epochs, device, history=None)
    
    def evaluate(model, dataloader):
        random = np.random.randint(0, len(dataloader))
        with torch.no_grad():
            model.eval()
            n_correct = 0
            n_samples = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, 1), 1)
            
            n_samples += labels.shape[0]
            n_correct += (preds==labels).sum().item()
            acc = 100.0 * n_correct / n_samples
            print(acc)
            
    evaluate(model,test_loader)

             
         
         