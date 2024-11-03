import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_prep import prepare_data, SRDataset
from torchvision import transforms

#directories
train_high_res_dir = "./archive/MineCraft-RT_1280x720_v14/MineCraft-RT_1280x720_v14/images"
val_high_res_dir = "./archive/MineCraft-RT_1280x720_v12/MineCraft-RT_1280x720_v12/images"
working_dir = './outputs'

#prepare data
prepare_data(train_high_res_dir, val_high_res_dir, working_dir)

#create transforms for images
transform = transforms.Compose([
    transforms.ToTensor()
])

#create datasets
train_low_res_dir = os.path.join(working_dir, 'train', 'inputs')
val_low_res_dir = os.path.join(working_dir, 'val', 'inputs')

train_dataset = SRDataset(train_low_res_dir, train_high_res_dir, transform=transform)
val_dataset = SRDataset(val_low_res_dir, val_high_res_dir, transform=transform)

print("\nNumber of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))

#create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

#define the SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        #adjust in_channels and out_channels for RGB images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

#initialize the model, loss function, and optimizer
model = SRCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

    #validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

#save the trained model
torch.save(model.state_dict(), 'srcnn_model.pth')
print("Model saved as 'srcnn_model.pth'")
