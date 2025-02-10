import torch
import torch.nn as nn
from torchvision import models

# First recreate your model architecture
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, 2)
)

# Load your trained weights
model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
model.eval()

# Export to TorchScript
example = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example)

# Save the model in TorchScript format
traced_model.save("pneumonia_model_mobile.pt")