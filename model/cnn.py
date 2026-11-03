import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

train_set = torchvision.datasets.ImageFolder(TRAIN_PATH, transform=transform)
test_set = torchvision.datasets.ImageFolder(TEST_PATH, transform=transform)

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