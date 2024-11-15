import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Define the number of classes and class names
class_names = ['normal', 'adenocarcinoma_left.lower', 'large.cell.carcinoma_left', 'squamous.cell.carcinoma_left'] 
number_of_classes = len(class_names)

# Initialize the ResNet-18 model pre-trained on ImageNet
resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Modify the final layer to match the number of classes you are classifying
resnet18_model.fc = nn.Sequential(
    nn.Linear(resnet18_model.fc.in_features, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Linear(256, number_of_classes)
)

# Check if a GPU is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18_model = resnet18_model.to(device)

# Function to load an image and transform it
def load_and_transform_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Function to predict the cancer type
def predict_cancer_type(model, image_path, class_names, device):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to match the input size of the model
        transforms.ToTensor()           # Transform it to a tensor
    ])

    # Load and transform the image
    image = load_and_transform_image(image_path, transform)
    image = image.to(device)  # Move the image to the appropriate device

    # Set the model to evaluation mode
    model.eval()

    # No need to track gradients for validation
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        predicted_class = class_names[predicted.item()]  # Retrieve the class name

    return predicted_class

# Ask the user for the image path
image_path = input("Please enter the path to the image you want to analyze: ")

# Example usage
model_path = 'models/ct_scan_model.pth'

# Load the model
model = resnet18_model  # Ensure the architecture matches the saved model
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Predict the cancer type
cancer_type = predict_cancer_type(model, image_path, class_names, device)
print("Predicted Cancer Type:", cancer_type)
