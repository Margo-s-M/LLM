import torch
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn
import torch.nn.functional as F
import numpy as np
import PIL.Image

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        for parameter in resnet.parameters():
            parameter.requires_grad = False

        resnet.fc = nn.Identity()
        self.resnet = resnet

        self.linear = nn.Linear(2048, 2)

    def forward(self, x):
        out = self.resnet(x)
        out = self.linear(out)

        return out


model= MyModel()
model.load_state_dict(torch.load('data/model_weights.pth', weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
classes = ['bird', 'drone']
#прогноз
img = PIL.Image.open('data/000000000023.jpg')
img.show("title")
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)

with torch.no_grad():
    model = model.to('cpu')
    logits = model(img_tensor)

proba = F.softmax(logits)

idx = proba.argmax()

print(logits)
print(idx)
print(proba)
