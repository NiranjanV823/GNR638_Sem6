#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image
from torchsummary import summary


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


device


# In[4]:


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        out2 = self.conv2(out)
        out2 = self.relu(out2)
        out2 += out
        out2 = self.conv2(out)
        out2 = self.relu(out2)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = self.make_residual_blocks()
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False),
            nn.Tanh()
        )

    def make_residual_blocks(self):
        return nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

    def forward(self, x):
        out = self.downsample(x)
        out = self.relu(out)
        out = self.residual_blocks(out)
        out = self.upsample(out)
        return out

# Test the model
generator = Generator().to(device)
input_tensor = torch.randn(1, 3, 256, 448).to(device)  # Batch size, channels, height, width
output_tensor = generator(input_tensor)
print("Output tensor shape:", output_tensor.shape)


summary(generator, (3, 256, 448))


# In[ ]:


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input: (3, 256, 448)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 128, 224)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 64, 112)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 32, 56)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),  # Output: (512, 31, 55)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # Output: (1, 30, 54)
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         out = self.layers(x)
#         return out

# # Test the model
# discriminator = Discriminator()
# input_tensor = torch.randn(1, 3, 256, 448)  # Batch size, channels, height, width

# output_tensor = discriminator(input_tensor)
# print("Output tensor shape:", output_tensor.shape)


# In[5]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
 
        self.model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        # nn.ZeroPad2d((0, 1, 0, 1)),
        nn.BatchNorm2d(64, momentum=0.82),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128, momentum=0.82),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256, momentum=0.8),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256, momentum=0.8),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256, momentum=0.8),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(256 * 4 * 7, 500),
        nn.Sigmoid(),
        # nn.Linear(100, 100),
        # nn.Sigmoid(),
        nn.Linear(500, 1),
        nn.Sigmoid()
    )
 
    def forward(self, img):
        validity = self.model(img)
        return validity

discriminator = Discriminator().to(device)
input_tensor = torch.randn(1, 3, 256, 448).to(device)
print(summary(discriminator,(3, 256, 448)))
  
output_tensor = discriminator(input_tensor)
print("Output tensor shape:", output_tensor.shape)


# In[6]:


from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_image_path = self.dataframe.iloc[idx, 0]
        final_image_path = self.dataframe.iloc[idx, 1]
        input_image = Image.open(input_image_path)
        final_image = Image.open(final_image_path)
        if self.transform:
            input_image = self.transform(input_image)
            final_image = self.transform(final_image)
        return input_image, final_image

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Dataset
dataframe = pd.read_csv("paths.csv")
dataset = CustomDataset(dataframe, transform=transform)

# DataLoader
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 


# In[7]:


# Define Loss Function and Optimizers
criterion = nn.BCELoss()
generator = Generator().to(device) 
discriminator = Discriminator().to(device) 
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)) 
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# In[8]:


# Training Loop
num_epochs = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator.to(device)
# discriminator.to(device)

for epoch in range(num_epochs):
    for i, (input_images, real_images) in enumerate(dataloader):
        batch_size = input_images.size(0)
        # print(batch_size)

        
        # Training Discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train with real images 
        real_images = real_images.to(device)
        # print(real_images.shape)
        real_outputs = discriminator(real_images) 
        real_loss = criterion(real_outputs, real_labels) 
 
        # Train with fake images 
        fake_images = generator(input_images.to(device)).detach()
        fake_outputs = discriminator(fake_images)
        fake_loss = criterion(fake_outputs, fake_labels)

        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Training Generator
        generator.zero_grad()
        fake_images = generator(input_images.to(device))
        outputs = discriminator(fake_images)
        generator_loss = criterion(outputs, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"Generator Loss: {generator_loss.item():.4f}, "
                  f"Discriminator Loss: {discriminator_loss.item():.4f}")


# !nvidia-smi

# In[ ]:





# In[10]:


print(torch.cuda.memory_summary())


# In[ ]:




