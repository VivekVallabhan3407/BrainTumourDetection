# BrainTumourDetection

## This project is a web-based application for detecting brain tumors using a Convolutional Neural Network (CNN) model. The application allows users to upload MRI images and predicts the type of tumor (if any) with confidence scores.

## Features
- ### **Upload MRI Images**: Users can upload MRI scans in .jpg, .jpeg, or .png formats.
- ### **Tumor Type Prediction**: The app predicts the type of brain tumor (e.g., glioma, meningioma, pituitary tumor, or no tumor).
- ### **Confidence Score**: Displays the confidence level of the prediction.
- ### **User-Friendly Interface**: A clean and intuitive interface built with Streamlit.

## Technologies Used
- ### **Frontend**: Streamlit for the web interface.
- ### **Backend**: TensorFlow and Keras for the deep learning model.
- ### **Model**: Pre-trained ResNet50 for feature extraction and a custom dense layer for classification.
- ### **Languages**: Python.

## Installation
Clone the repository:
 ```bash
   git clone https://github.com/your-username/BrainTumorDetection.git
   cd BrainTumorDetection
```
Create a virtual environment and activate it:
```python -m venv tumor_env
tumor_env\Scripts\activate  # On Windows
```
Install the required dependencies:
```pip install -r requirements.txt```
Place the trained model file (resnet_dense_model_30_epochs (1).h5) in the models directory.

## Usage
- Run the application:
- ```streamlit run src/app.py```
- Open the application in your browser at http://localhost:8501.
- Upload an MRI image and view the prediction results.

## Project Structure
BRAIN_TUMOUR/
├── models/
│   └── resnet_dense_model_30_epochs (1).h5
├── src/
│   └── app.py
├── test_images/
│   ├── glioma.png
│   ├── pituitary.png
│   ├── notumor.png
│   └── meningioma.png
├── tumor_env/
│   ├── etc
│   ├── Include
│   ├── Lib
│   ├── Scripts
│   ├── share
│   └── pyvenv.cfg
└── requirements.txt

## Screenshots
1. Interface Without Upload
![Screenshot 2025-05-09 103232](https://github.com/user-attachments/assets/496424a0-a20a-4624-b336-c1bf808f45c6)


2. Interface With Uploaded Image
![Screenshot 2025-05-09 103426](https://github.com/user-attachments/assets/a4ac0090-1721-4595-aaac-7901bdf2ea5d)


## Disclaimer
### This application is for educational purposes only. The predictions are not a substitute for professional medical advice. Please consult a doctor for an accurate diagnosis.

## License
### This project is licensed under the Apache License 2.0.

## Acknowledgments
The ResNet50 model is pre-trained on the ImageNet dataset.
Streamlit for providing an easy-to-use framework for building web applications.
