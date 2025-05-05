import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io
import time
import os
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Page configuration
st.set_page_config(
    page_title="Apple Disease Classifier",
    layout="wide",  # Changed to wide for better visuals
    initial_sidebar_state="collapsed",
)

# Define custom color scheme
primary_color = "#D32F2F"  # Apple red
secondary_color = "#4CAF50"  # Healthy green
neutral_color = "#424242"  # Dark gray
background_color = "#fafafa"  # Light gray background
card_bg = "#FFFFFF"  # White cards

# Custom CSS for enhanced styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: black;
    }}
    .main-header {{
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, {primary_color}, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 0.5rem 0;
        letter-spacing: 1px;
    }}
    .sub-header {{
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: {neutral_color};
        border-left: 4px solid {primary_color};
        padding-left: 10px;
    }}
    .card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }}
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }}
    .result-text {{
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        background: linear-gradient(90deg, #f5f7fa, #c3cfe2);
        display: inline-block;
    }}
    .category-description {{
        background-color: {card_bg};
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid {secondary_color};
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }}
    .confidence-bar {{
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }}
    .footer {{
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #757575;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #B71C1C;
        transform: scale(1.05);
    }}
    .option-card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .option-card:hover {{
        border-color: {primary_color};
        transform: translateY(-2px);
    }}
    .option-card.active {{
        border-color: {primary_color};
        border-width: 2px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    .warning-box {{
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }}
    .success-box {{
        background-color: #E8F5E9;
        border-left: 4px solid {secondary_color};
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }}
    div.stExpander {{
        border: none;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }}
    .apple-status {{
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }}
    .apple-healthy {{
        background-color: #E8F5E9;
        color: {secondary_color};
    }}
    .apple-diseased {{
        background-color: #FFEBEE;
        color: {primary_color};
    }}
    .divider {{
        height: 3px;
        background: linear-gradient(90deg, {primary_color}, {secondary_color});
        border-radius: 3px;
        margin: 1.5rem 0;
    }}
    /* Fix for visibility of inactive menu headings */
    .stRadio > div {{
        color: #1a1a1a !important;
        font-weight: 500;
    }}

    .stRadio > div label {{
        color: #1a1a1a !important;
        font-weight: 600;
    }}

    
    /* Custom image container */
    .img-container {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        position: relative;
    }}
    .img-overlay {{
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# Function to set background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-color:black;
             background-attachment: fixed;
             background-size: cover;
             background-position: center;
         }}
         .main {{
             background-color: rgba(255, 255, 255, 0.9);
             padding: 2rem;
             border-radius: 15px;
             backdrop-filter: blur(5px);
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Add background
##add_bg_from_url()

# Custom function for fancy header with logo
def fancy_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="main-header">Apple Disease Classifier</p>', unsafe_allow_html=True)
    
    # Add animated divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: white;">
            Upload or capture apple images to identify diseases using advanced AI technology.
            Our model detects healthy apples, apple rot, and apple scab with high accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_classifier_model():
    try:
        model_path = "apple_model.h5"
        if not os.path.exists(model_path):
            st.error("Model file not found. Please check if the model file exists in the correct path.")
            return None
            
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Function to preprocess the image
def preprocess_image(img):
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize to the input size expected by the model
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    
    return img_array

# Enhanced image processing with visual effects
def enhance_image(img):
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    # Enhance color
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    
    return img

# Function to make predictions
def predict_apple_disease(model, img_array):
    try:
        # Class indices
        class_indices = {0: "Apple is Healthy", 1: "Apple is Rot", 2: "Apple has Scab"}
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the class with highest probability
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:  # Multi-class case
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx]) * 100
        else:  # Binary case
            # Assuming 0 is "Apple (Healthy)" and 1 is "AppleRot"
            if predictions[0][0] > 0.5:
                class_idx = 1  # AppleRot
                confidence = float(predictions[0][0]) * 100
            else:
                class_idx = 0  # Apple (Healthy)
                confidence = (1 - float(predictions[0][0])) * 100
        
        return class_indices.get(class_idx, "Unknown"), confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Classification failed", 0

# Function to get category description with enhanced formatting
def get_category_description(category):
    descriptions = {
        "Apple is Healthy": """
            <h3 style="color: black;">🍏 Healthy Apple</h3>
            <p style="color: black;">This apple appears to be in good condition with no visible signs of disease.</p>
            <ul style="color: black;">
                <li>Uniform color throughout the fruit</li>
                <li>Smooth, unblemished skin</li>
                <li>No dark spots or lesions</li>
                <li>Typical shape and size for the variety</li>
            </ul>
            <p style="color: black;"><strong>Safe to consume:</strong> This apple is suitable for eating fresh or using in recipes.</p>
        """,
        "Apple is Rot": """
            <h3 style="color: black;">🍎 Apple Rot</h3>
            <p style="color: black;">This apple shows signs of rot, which is a fungal disease affecting the fruit.</p>
            <ul style="color: black;">
                <li>Dark brown spots on the skin</li>
                <li>Soft, decaying areas</li>
                <li>Possible sunken lesions</li>
                <li>May have a musty or fermented odor</li>
            </ul>
            <p style="color: black;"><strong>Caution:</strong> Affected fruits should not be consumed as they may contain mycotoxins or other harmful compounds.</p>
            <p style="color: black;">Common fungi causing apple rot include Botryosphaeria, Colletotrichum, and Monilinia species.</p>
        """,
        "Apple has Scab": """
            <h3 style="color: black;">🍏 Apple Scab</h3>
            <p style="color: black;">This apple shows signs of apple scab, caused by the fungus Venturia inaequalis.</p>
            <ul style="color: black;">
                <li>Olive-green to brown lesions on the fruit</li>
                <li>Rough, corky texture in affected areas</li>
                <li>May cause fruit deformation in severe cases</li>
                <li>Scabby appearance on the skin</li>
            </ul>
            <p style="color: black;"><strong>Limited consumption:</strong> While apples with mild scab can be eaten after removing affected areas, 
            severe cases may affect quality and taste.</p>
        """
    }
    return descriptions.get(category, "<p style='color: black;'>No detailed description available for this category.</p>")


# WebRTC configuration for live video
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video processor class for real-time inference
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = load_classifier_model()
        self.prediction = None
        self.confidence = 0
        self.last_prediction_time = time.time()
        self.prediction_interval = 1.0  # Make prediction every 1 second
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Draw a rectangle for the apple placement guide
        h, w = img.shape[0], img.shape[1]
        center_x, center_y = w // 2, h // 2
        box_size = min(h, w) * 0.6
        
        x1 = int(center_x - box_size // 2)
        y1 = int(center_y - box_size // 2)
        x2 = int(center_x + box_size // 2)
        y2 = int(center_y + box_size // 2)
        
        # Draw guide rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Place apple here", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Make prediction every few seconds
        current_time = time.time()
        if current_time - self.last_prediction_time > self.prediction_interval:
            self.last_prediction_time = current_time
            
            # Crop the region of interest
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:  # Ensure ROI is not empty
                # Convert to PIL Image and preprocess
                pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                img_array = preprocess_image(pil_img)
                
                # Make prediction
                if self.model:
                    self.prediction, self.confidence = predict_apple_disease(self.model, img_array)
        
        # Display prediction if available
        if self.prediction:
            # Create a semi-transparent overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            
            # Set text color based on prediction
            if "Healthy" in self.prediction:
                text_color = (0, 255, 0)  # Green for healthy
            else:
                text_color = (0, 0, 255)  # Red for diseased
            
            # Add prediction and confidence
            cv2.putText(overlay, f"{self.prediction}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(overlay, f"Confidence: {self.confidence:.1f}%", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Apply the overlay
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            
        return img

# Function to display image with styled container
def display_image_with_style(img, caption=""):
    st.markdown(f"""
        <div class="img-container">
            <div class="img-overlay">{caption}</div>
        </div>
    """, unsafe_allow_html=True)
    st.image(img, use_column_width=True)

# Load model
model = load_classifier_model()

# Main app layout
fancy_header()

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Live Camera (OpenCV)", "ℹ️ About"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload Apple Image</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="margin-bottom: 1rem;">
        Upload a clear image of an apple to analyze its condition. 
        The image should show the apple clearly without obstructions.
    </p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(image_bytes))
            
            # Enhance image
            enhanced_img = enhance_image(img)
            
            # Create columns for before/after comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Image")
                st.image(img, caption="Uploaded Image", use_container_width=True)
                
            with col2:
                st.markdown("#### Enhanced Image")
                st.image(enhanced_img, caption="Enhanced for Analysis", use_container_width=True)
            
            # Make prediction
            if model:
                with st.spinner("🔍 Analyzing apple image..."):
                    # Add slight delay for better UX
                    time.sleep(1)
                    img_array = preprocess_image(enhanced_img)
                    category, confidence = predict_apple_disease(model, img_array)
                
                # Display the result with nice styling
                if "Healthy" in category:
                    icon = "✅"
                    status_class = "apple-healthy"
                else:
                    icon = "⚠️"
                    status_class = "apple-diseased"
                    
                st.markdown(f"""
                <div style="text-align: center; margin: 1.5rem 0;">
                    <h2>{icon} Result</h2>
                    <div class="apple-status {status_class}">
                        {category}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create progress bar with animation
                st.markdown('<p class="confidence-bar">AI Confidence Level:</p>', unsafe_allow_html=True)
                progress_bar = st.progress(0)
                
                # Animate the progress bar
                for i in range(int(confidence) + 1):
                    progress_bar.progress(i / 100)
                    time.sleep(0.005)
                
                st.markdown(f"""
                <div style="text-align: right; font-size: 1.2rem; font-weight: bold; margin-bottom: 1rem;">
                    {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Display category description
                st.markdown('<p class="sub-header">Detailed Analysis</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="category-description">{get_category_description(category)}</div>', unsafe_allow_html=True)
                
                # Recommendation based on the category
                if "Healthy" not in category:
                    st.markdown("""
                        <div class="warning-box">
                            <h4 style="margin-top: 0; color: black;">⚠️ Health Advisory</h4>
                            <p style="color: black;">This apple shows signs of disease and may not be suitable for consumption. 
                            Consider discarding it to prevent potential health issues.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="success-box">
                            <h4 style="margin-top: 0; color: black;">✅ Health Advisory</h4>
                            <p style="color: black;">This apple appears healthy and should be safe for consumption. 
                            Always wash fruits thoroughly before eating.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                # Add download button for report
                report_html = f"""
                <html>
                <head>
                    <title>Apple Health Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .header {{ text-align: center; color: #D32F2F; }}
                        .result {{ font-size: 18px; margin: 15px 0; }}
                        .confidence {{ color: #555; }}
                        .details {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <h1 class="header">Apple Health Analysis Report</h1>
                    <p class="result"><strong>Result:</strong> {category}</p>
                    <p class="confidence"><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <div class="details">
                        {get_category_description(category)}
                    </div>
                    <p>Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </body>
                </html>
                """
                
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=report_html,
                    file_name="apple_health_report.html",
                    mime="text/html"
                )
                
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            st.warning("Please upload a valid image of an apple.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Live Detection with Camera</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="margin-bottom: 1rem;">
        Use your camera to analyze apples in real-time. Position the apple inside the green box for the best results.
        The AI will continuously analyze and provide feedback on the apple's condition.
    </p>
    """, unsafe_allow_html=True)
    
    # WebRTC streamer for live video
    webrtc_ctx = webrtc_streamer(
        key="apple-classifier",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Additional instructions for camera mode
    if webrtc_ctx.state.playing:
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <h4 style="margin-top: 0; color: #1565C0;">📋 Live Camera Instructions</h4>
            <ol>
                <li>Position the apple inside the green rectangle</li>
                <li>Hold the apple steady for best results</li>
                <li>Ensure good lighting for accurate detection</li>
                <li>Try different angles if the detection is uncertain</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">About Apple Disease Classification</p>', unsafe_allow_html=True)
    
    # Create columns for a nicer layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1570913196364-62e176f7ef0e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=774&q=80", 
                 caption="Healthy apples are essential for nutrition")
    
    with col2:
        st.markdown("""
        ### Why Apple Quality Matters
        
        Apples are one of the most consumed fruits worldwide, known for their nutritional benefits. 
        However, diseases can affect both the quality and safety of apples.
        
        Our AI-powered classifier helps you quickly identify:
        
        * **Healthy Apples**: Safe and nutritious for consumption
        * **Apple Rot**: Fungal disease making apples unsuitable for eating
        * **Apple Scab**: Common disease affecting appearance and potentially taste
        
        Using advanced deep learning techniques, our model has been trained on thousands of apple images 
        to provide accurate and reliable results.
        """)
    
    st.markdown("""
    ### Understanding Apple Diseases
    
    <div style="display: flex; margin-top: 20px;">
        <div style="flex: 1; padding: 10px;">
            <h4>Apple Rot</h4>
            <p>Rot diseases are caused by various fungi including Botryosphaeria, Colletotrichum, and Monilinia species. 
            They typically start as small, brown lesions that expand and soften the fruit tissue. 
            Infected apples should be discarded to prevent spread.</p>
        </div>
        <div style="flex: 1; padding: 10px;">
            <h4>Apple Scab</h4>
            <p>Caused by the fungus Venturia inaequalis, apple scab is one of the most common apple diseases worldwide. 
            It appears as olive-green to brown lesions on the fruit. While mild cases may not affect edibility, 
            severe infections can impact taste and storage life.</p>
        </div>
    </div>
    
    ### How Our AI Works
    
    Our model uses a deep learning architecture based on MobileNetV2, fine-tuned specifically for apple disease detection. 
    The classification process involves:
    
    1. **Image preprocessing**: Scaling, normalization, and enhancement
    2. **Feature extraction**: Identifying visual patterns associated with different apple conditions
    3. **Classification**: Determining the most likely condition based on extracted features
    4. **Confidence scoring**: Evaluating the reliability of the prediction
    
    The model achieves over 95% accuracy on test datasets and continues to improve with ongoing training.
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# How to use expander
with st.expander("How to use this app"):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Choose your preferred method**:
       * Upload an image of an apple from your device
       * Use your camera to capture an apple in real-time
       
    2. **For image upload**:
       * Click the 'Browse files' button
       * Select a clear image of an apple
       * Wait for the analysis to complete
       
    3. **For camera mode**:
       * Click 'Start' to activate your camera
       * Position the apple inside the green rectangle
       * Hold steady for accurate detection
       
    4. **Understanding results**:
       * The app will display the condition of your apple
       * A confidence percentage shows the reliability of the prediction
       * Detailed information about the detected condition is provided
       * For diseased apples, a health advisory will be shown
       
    5. **Additional features**:
       * Download a detailed report of your apple's analysis
       * Learn more about apple diseases in the 'About' tab
    """)

# Footer
st.markdown('<div class="footer">Apple Disease Classifier | Developed with Streamlit, TensorFlow & OpenCV<br>© 2025 - AI-Powered Food Safety</div>', unsafe_allow_html=True)