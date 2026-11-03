import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from main import drawBoundingBox, extractHands, detector
import cv2


def cropToHand(frame, landmarks):
    height, width, _ = frame.shape
    x_min, y_min, x_max, y_max = drawBoundingBox(landmarks, width, height)
    crop = frame[y_min:y_max, x_min:x_max]
    if (crop.size > 0):
        return crop
    else:
        return None

def predict(frame_bgr, landmarks):
    crop = cropToHand(frame_bgr, landmarks)
    if crop is None:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        return model(x).argmax(1).item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
WEIGHT_DECAY = 1e-4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, "data", "Train")
TEST_PATH = os.path.join(ROOT, "data", "Test")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




class croppedHandsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.base = torchvision.datasets.ImageFolder(root)
        self.transform = transform
        self.samples = []
        for i in range(len(self.base)):
            img, label = self.base[i]
            frame = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            hands = extractHands(frame, detector)
            if not hands:
                continue
            crop = cropToHand(frame, hands[0]["landmarks"])
            if crop is None:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            self.samples.append((Image.fromarray(crop), label))
        self.classes = self.base.classes
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, label = self.samples[i]
        return self.transform(img), label




train_set = croppedHandsDataset(TRAIN_PATH, transform)
test_set = croppedHandsDataset(TEST_PATH, transform)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32,64,kernel_size = 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*8*8,512)
        self.fc2 = nn.Linear(512,24)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR,betas=(0.9,0.999),eps=1e-08,weight_decay=WEIGHT_DECAY)



for epoch in range(EPOCHS):
    model.train()
    runningLoss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        runningLoss += loss.item() * images.size(0)

    epoch_loss = runningLoss 
    print("Epoch ", epoch, "of ", EPOCHS)
    print("Loss: ", epoch_loss)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: ", 100*correct/total)