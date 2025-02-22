import torch
import os
from PIL import Image
from torchvision import transforms

# Define paths
model_path = "xview2_model.pth"  # Ensure this file is in the same directory
input_folder = ""  # Change this to your actual image folder
output_folder = ""  # Output folder for predictions

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the pre-trained xBD model
print("📥 Loading model...")
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to predict damage
def predict_damage(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return prediction

# Process all images in the input folder
print("🔍 Running predictions on images...")
for image_file in os.listdir(input_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, image_file)
        prediction = predict_damage(image_path)

        # Save the result
        result_file = os.path.join(output_folder, f"{image_file}_prediction.txt")
        with open(result_file, "w") as f:
            f.write(f"Prediction for {image_file}: {prediction}\n")

        print(f"✅ Processed {image_file}: Damage Level = {prediction}")

print("🎉 All predictions completed!")