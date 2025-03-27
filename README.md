# Dog Image Classifier

A StreamLit application that uses a pre-trained ResNet50 model to classify whether an uploaded image contains a dog or not.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the StreamLit application:
```bash
streamlit run app.py
```

## Usage

1. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
2. Click the "Choose an image..." button to upload an image
3. Click the "Classify Image" button to analyze the image
4. The application will tell you whether the image contains a dog or not, along with the predicted class

## Features

- Uses pre-trained ResNet50 model for accurate image classification
- Supports common image formats (JPG, JPEG, PNG)
- Real-time image processing and classification
- User-friendly interface with clear results 