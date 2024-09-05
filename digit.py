import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_df = pd.read_csv('train.csv')
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

# test train split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


test_df = pd.read_csv('test.csv')
X_test = test_df

class DigitDataset(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X.iloc[idx].values.astype(np.uint8).reshape(28, 28, 1)
        if self.transform:
            X = self.transform(X)
        if self.y is not None:
            return X, self.y.iloc[idx]
        return X

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1))
])

train_dataset = DigitDataset(X_train, y_train, transform)
test_dataset = DigitDataset(X_test, transform=transforms.ToTensor())
val_dataset = DigitDataset(X_val, y_val, transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lr = 0.001, params=model.parameters(), weight_decay=0.0001, amsgrad=True) 


for epoch in tqdm(range(len(train_loader))):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    if loss.item() < 0.01:
        break

model.eval()

# validation set
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        _, predicted = torch.max(y_pred.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f'Accuracy: {correct / total}')

# test set
preds = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        _, predicted = torch.max(y_pred.data, 1)
        preds.extend(predicted.cpu().numpy())

submission = pd.DataFrame({'ImageId': range(1, len(preds) + 1), 'Label': preds})
submission.to_csv('submission.csv', index=False)

# save model
torch.save(model.state_dict(), 'model.pth')
