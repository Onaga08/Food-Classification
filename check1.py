import torch
from torchvision import transforms
from PIL import Image

# Define the path to your saved model
model_path = r"food_classification_model1.pt"

# Load the model
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Define a transform to preprocess the input image
# Make sure the transform is consistent with what was used during training
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load and preprocess the input image
image_path = r"food-101\images\apple_pie\134.jpg"
input_image = Image.open(image_path)
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Optionally, apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Print the predicted class index
predicted_class = torch.argmax(probabilities).item()
print("Predicted class:", predicted_class)
