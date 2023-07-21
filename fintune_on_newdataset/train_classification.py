import os, sys, glob, time, random, shutil, copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchsummary import summary
from matplotlib import pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights # do not import
from PIL import Image, ImageFile
from skimage import io
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the train and validation directory paths
train_directory = 'dataset/train'
valid_directory = 'dataset/val'

# Batch size
bs = 64 
# Number of epochs
num_epochs = 20
# Number of classes
num_classes = 4
# Number of workers
num_cpu = 8 

# Applying transforms to the data
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
 
# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}
 
# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=False),
    'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=False,
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)
 
# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

modelname = 'resnet18'

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=None)    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)


# Transfer the model to GPU
model = model.to(device)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model.named_parameters()):
    print(num, name, param.requires_grad )
summary(model, input_size=(3, 224, 224))

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer 
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Learning rate decay
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(1, num_epochs+1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        n = 0
        stream = tqdm(dataloaders[phase])
        for i, (inputs, labels) in enumerate(stream, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            n += inputs.shape[0]
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            stream.set_description(f'Batch {i}/{len(dataloaders[phase])} | Loss: {running_loss/n:.4f}, Acc: {running_corrects/n:.4f}')

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('Epoch {}-{} Loss: {:.4f} Acc: {:.4f}'.format(
            epoch, phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'valid' and epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Update best model!')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
torch.save(model, 'logs/resnet18_4class.pth')
torch.save(model.state_dict(), 'logs/resnet18_4class.tar')
