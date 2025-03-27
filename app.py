import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Dog Image Classifier",
    page_icon="🐕",
    layout="centered"
)

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    return model

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class labels
@st.cache_data
def load_labels():
    with open('imagenet_classes.txt') as f:
        return [line.strip() for line in f.readlines()]

def is_dog_breed(label):
    # List of terms that indicate the image is of a dog
    dog_indicators = [
        'dog', 'retriever', 'spaniel', 'terrier', 'hound', 'poodle', 
        'shepherd', 'collie', 'husky', 'malamute', 'chihuahua', 
        'bulldog', 'mastiff', 'pug', 'corgi', 'beagle', 'rottweiler',
        'schnauzer', 'dachshund'
    ]
    return any(indicator in label.lower() for indicator in dog_indicators)

def main():
    st.title("🐕 Dog Image Classifier")
    st.write("Upload an image to check if it contains a dog!")

    # Load model and labels
    model = load_model()
    labels = load_labels()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add a classify button
        if st.button("Classify Image"):
            with st.spinner("Analyzing image..."):
                # Preprocess the image
                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(0)

                # Make prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_label = labels[predicted.item()]

                # Check if the prediction is a dog breed
                is_dog = is_dog_breed(predicted_label)
                
                # Display result with appropriate styling
                if is_dog:
                    st.success("Yes, this image contains a dog! 🐕")
                    st.write(f"Predicted class: {predicted_label}")
                else:
                    st.error("No, this image does not contain a dog.")
                    st.write(f"Predicted class: {predicted_label}")

if __name__ == "__main__":
    main() 