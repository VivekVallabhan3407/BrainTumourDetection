import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('./models/resnet_dense_model_30_epochs (1).h5')

resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

index_to_class = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}


st.set_page_config(page_title="Brain Tumor Detection", layout="wide")


st.markdown("""
    <style>
     .upload-box {
        width: 1000px;
        height: 500px;
        border: 4px dashed #4CAF50;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px auto;
        background-color: #f9f9f9;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }


    .image-inside-box {
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }
     .image-inside-box > div {
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

     .image-inside-box img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        margin: auto;
        padding: 10px;
    }
   

    .upload-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
    }


    .upload-text {
        font-size: 24px;
        color: #666;
    }

   
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1], gap="large")

with col2:
    st.write("")  # Empty space
    st.write("")  # Add more empty space if needed
    st.subheader("Disclaimer:")
    st.warning("This is just a prediction. For an accurate diagnosis, please consult a doctor.")

with col1:
    st.write("")  # Empty space
    st.markdown("<h1 style='text-align: center;'>Brain Tumor Detection</h1>", unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        # Show upload box placeholder
        st.markdown('''<div class="upload-box">
                <div class="upload-placeholder">
                    <div class="upload-text">Your MRI scan image will appear here</div>
                </div>
                </div>
            </div>''', unsafe_allow_html=True)
    else:
        import base64
        from io import BytesIO

        img = Image.open(uploaded_file).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Embed the image inside the upload box
        st.markdown(f'''
            <div class="upload-box">
                <div class="image-inside-box">
                    <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; max-height: 100%; object-fit: contain;" />
                </div>
            </div>
        ''', unsafe_allow_html=True)

    
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = resnet.predict(img_array)

        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)
        tumor_type = index_to_class[predicted_class]
        confidence = prediction[0][predicted_class] * 100  

        st.subheader("Predicted Tumor Type:")
        st.success(f"{tumor_type} (Confidence: {confidence:.2f}%)")
        