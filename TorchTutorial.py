import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
data = [[1,2,3,4],[3,4,5,6]]
x_data = torch.tensor(data) #turn matrix into tensor
x_rand = torch.rand_like(x_data, dtype=torch.float) #Convert int to random floating points
#print(x_rand)
#Matrix Multiplication
tensor1 = torch.randn(3,4) #Create a random tensor 3x4
tensor2 = torch.randn(4) #Create a random tensor 4
result = torch.matmul(tensor1, tensor2)
#print(result)
"""Image Classifier NN Model"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), #Flatten 28x28 image to 512 tensor
            nn.ReLU(), #Activation function
            nn.Linear(512, 512),
            nn.ReLU(), #Activation function
            nn.Linear(512, 10) #outputs 10 class names
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel (for 28x28)
    transforms.Resize((28, 28))  # Resize to 28x28 since your model expects that
])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
model = Net().to('cpu')
X, y = next(iter(train_loader))
X, y = X.to('cpu'), y.to('cpu')
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred_n = torch.argmax(pred_probab, dim=1)
print("The predicted value is", y_pred_n)
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
y_pred = y_pred_n.cpu().numpy()
X_cpu = X.cpu()
batch_accuracy = (y_pred == y.cpu()).sum().item() / len(y.cpu())
print(f"✅ Batch Accuracy: {batch_accuracy * 100:.2f}%")
# -----------------------------
# 🖼️ Show sample images with predicted labels
# -----------------------------
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= len(X_cpu):
        break
    img = X_cpu[i].permute(1, 2, 0).numpy()  # Convert from C×H×W → H×W×C
    ax.imshow(img.squeeze(), cmap="gray")    # Grayscale or RGB handled automatically
    ax.set_title(classes[y_pred[i]], fontsize=8)
    ax.axis("off")

plt.suptitle("Predicted Classes for Batch Images", fontsize=14)
plt.tight_layout()
plt.show()

# Count how many predictions per class in the batch
unique, counts = np.unique(y_pred, return_counts=True)

# Create bar plot
plt.figure(figsize=(8, 5))
plt.bar([classes[i] for i in unique], counts, color='skyblue', edgecolor='black')
plt.title("Predicted Class Distribution (Batch of 32)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()