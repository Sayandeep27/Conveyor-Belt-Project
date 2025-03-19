from flask import Flask, request, render_template
import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

app = Flask(__name__)

# Load Hugging Face model
MODEL_NAME = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

LABELS = ["Good", "Damaged"]  # Replace with actual labels if needed

def classify_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[predicted_class % len(LABELS)]  # Ensure index is valid

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file).convert("RGB")
            result = classify_image(image)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
