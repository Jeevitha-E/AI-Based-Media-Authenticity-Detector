import streamlit as st
import time
import os
import numpy as np
import tempfile
import json
import h5py
from datetime import datetime
from PIL import Image

# Machine Learning Imports
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import librosa
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import cv2
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="AI Media Validator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS & STYLING (COLORFUL & MODERN) =================
st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }

    /* Main Background - Professional Abstract */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1579546929518-9e396f3cc809?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Content Container */
    /* Minimalist card on white */
    .main .block-container {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08); /* Soft shadow for depth */
        border: 1px solid #f0f0f0;
        margin: 2rem auto;
        width: 90%; /* Default for mobile */
        max-width: 1000px;
    }

    /* Tablet breakpoint */
    @media (min-width: 768px) {
        .main .block-container {
            width: 80%;
            padding: 3rem;
        }
    }

    /* Desktop breakpoint */
    @media (min-width: 1024px) {
        .main .block-container {
            width: 60%;
            padding: 4rem;
        }
    }

    /* Headings */
    h1 {
        color: #1E88E5;  /* Professional Blue */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 800;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
    }
    
    @media (min-width: 768px) {
        h1 { font-size: 2.5rem; }
    }
    
    h2, h3 {
        color: #424242; /* Dark Grey for subheaders */
        border-bottom: 2px solid #E3F2FD; /* Light Blue Accent */
        padding-bottom: 10px;
        font-size: 1.2rem;
    }
    
    @media (min-width: 768px) {
        h2, h3 { font-size: 1.5rem; }
    }

    /* Buttons */
    .stButton > button {
        background-image: linear-gradient(to right, #2196F3 0%, #1976D2 100%); /* Material Blue */
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(33, 150, 243, 0.3);
        width: 100%; /* Full width on mobile */
    }
    
    @media (min-width: 768px) {
        .stButton > button {
            width: auto;
            padding: 0.5rem 2rem;
        }
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(33, 150, 243, 0.4);
        color: white;
    }

    /* Auth Forms */
    [data-testid="stForm"] {
        background: #FAFAFA; /* Very light grey */
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #EEEEEE;
        box-shadow: inset 0 0 5px rgba(0,0,0,0.02);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #EEEEEE;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #ddd;
    }

    /* Result Cards */
    .result-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 6px solid #2196F3;
        border: 1px solid #f0f0f0; /* Add border for definition */
        border-left-width: 6px; /* Restore left accent */
        margin-top: 1rem;
        overflow-x: auto;
    }
    
    .verdict-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    
    .authentic {
        background: linear-gradient(to right, #43A047, #66BB6A); /* Material Green */
    }
    
    .fake {
        background: linear-gradient(to right, #D32F2F, #EF5350); /* Material Red */
    }

</style>
""", unsafe_allow_html=True)

# ================= COMPATIBILITY LAYER =================
if hasattr(st, 'cache_resource'):
    cache_resource = st.cache_resource
else:
    def cache_resource(func):
        return st.cache(allow_output_mutation=True)(func)

if hasattr(st, 'rerun'):
    def rerun():
        st.rerun()
else:
    def rerun():
        st.experimental_rerun()

# ================= CUSTOM KERAS LAYER (GLOBAL PATCH) =================
# AGGRESSIVE FIX: Monkeypatch the global InputLayer to handle 'batch_shape'
# This is necessary because load_model(custom_objects=...) might not apply to internal config deserialization 
# if the class name matches a standard keras layer.

OriginalInputLayer = tf.keras.layers.InputLayer

class PatchedInputLayer(OriginalInputLayer):
    def __init__(self, *args, **kwargs):
        # Keras 3 uses 'batch_shape', Keras 2 uses 'batch_input_shape'
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return cls(**config)

# Apply Patch Globally
tf.keras.layers.InputLayer = PatchedInputLayer

# ================= SESSION STATE =================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'users_db' not in st.session_state:
    st.session_state.users_db = {'admin': 'admin123'}

# ================= MODEL LOADING FUNCTIONS =================
@cache_resource
def load_audio_model():
    try:
        model_path = "audio_detector.h5"
        if not os.path.exists(model_path):
            return None, f"Audio model not found at {model_path}"
        
        # METHOD 1: Try Standard Load
        try:
            return tf.keras.models.load_model(model_path), None
        except:
            pass

        # METHOD 2: Advanced Keras 3 -> Keras 2 Config Downgrade
        # The user has a Keras 3 model (with DTypePolicy and batch_shape)
        # but is running Keras 2. We must recursively clean the config.
        
        def fix_layer_config(config):
            """Recursively fixes Keras 3 config incompatibilities for Keras 2."""
            if isinstance(config, dict):
                # Fix 1: 'batch_shape' -> 'batch_input_shape'
                if 'batch_shape' in config:
                    config['batch_input_shape'] = config.pop('batch_shape')
                
                # Fix 2: 'dtype' policy dict -> simple string
                if 'dtype' in config and isinstance(config['dtype'], dict):
                    # Keras 3: {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}}
                    # Keras 2: 'float32'
                    if 'config' in config['dtype'] and 'name' in config['dtype']['config']:
                         config['dtype'] = config['dtype']['config']['name']
                    else:
                        # Fallback default
                        config['dtype'] = 'float32'

                # Recurse into all values
                for key, value in config.items():
                   config[key] = fix_layer_config(value)
            
            elif isinstance(config, list):
                return [fix_layer_config(item) for item in config]
                
            return config

        try:
            with h5py.File(model_path, 'r') as f:
                if 'model_config' not in f.attrs:
                    raise ValueError("No config found in H5")
                config_str = f.attrs.get('model_config')
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
            
            # Load config as dict, clean it, dump back to string
            model_config = json.loads(config_str)
            model_config = fix_layer_config(model_config)
            
            # Reconstruct
            model = model_from_json(json.dumps(model_config))
            model.load_weights(model_path)
            return model, None
        except Exception as e:
            return None, f"Failed to patch and load model (DType/Batch Fix): {e}"

    except Exception as e:
        return None, str(e)

@cache_resource
def load_image_model():
    try:
        model_path = "Vision_Transformer_Model.pth"
        if not os.path.exists(model_path):
            return None, f"Image model not found at {model_path}", None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RGBFFT_ViT(num_classes=2).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, None, device
    except Exception as e:
        return None, str(e), None

@cache_resource
def load_video_model():
    try:
        model_path = "best.pt"
        if not os.path.exists(model_path):
            return None, f"Video model not found at {model_path}"
        # Prevent YOLO from printing
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

# ================= PROCESSING HELPERS =================
def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0

# Deepfake Audio Preprocessing (Matching User Logic)
# Deepfake Audio Preprocessing (Matching User Logic)
def preprocess_audio(file_path, sr=22050, n_mels=128, max_len=87):
    # Removed try-except to allow errors to propagate to the UI
    y, sr = librosa.load(file_path, sr=sr)
    
    # trim silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # amplitude normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
        
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # FORCE EXACT SHAPE USED IN TRAINING
    if mel_db.shape[1] < max_len:
        padding = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, padding)))
    else:
        mel_db = mel_db[:, :max_len]
        
    return mel_db

# Image Preprocessing
def fft_feature(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
normalize_6ch = transforms.Normalize(
    mean=[0.485, 0.485, 0.485, 0.485, 0.485, 0.485],
    std=[0.229, 0.229, 0.229, 0.229, 0.229, 0.229]
)

class RGBFFT_ViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, in_chans=6, num_classes=0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# ================= UI PAGES =================
def login_page():
    # Centered Login Card
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h1>🔐 WELCOME BACK</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#555;'>Sign in to access AI detection tools</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("SIGN IN")
            
            if submit:
                if username in st.session_state.users_db and st.session_state.users_db[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.user = username
                    st.success("Login successful! Redirecting...")
                    time.sleep(0.5)
                    rerun()
                else:
                    st.error("Invalid credentials. Try 'admin' / 'admin123'")
        
        st.markdown("<div style='text-align: center; margin-top: 1rem;'>New here?</div>", unsafe_allow_html=True)
        if st.button("Create New Account"):
            st.session_state.page = "signup"
            rerun()

def signup_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h1>🚀 JOIN THE REVOLUTION</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#555;'>Create an account to verify media authenticity</p>", unsafe_allow_html=True)
        
        with st.form("signup_form"):
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("CREATE ACCOUNT")
            
            if submit:
                if new_user in st.session_state.users_db:
                    st.error("Username already exists")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif not new_user or not new_pass:
                    st.error("All fields are required")
                else:
                    st.session_state.users_db[new_user] = new_pass
                    st.success("Account created! Please log in.")
                    time.sleep(1)
                    st.session_state.page = "login"
                    rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Login"):
            st.session_state.page = "login"
            rerun()

def main_app():
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        st.write(f"Logged in as: **{st.session_state.user}**")
        st.markdown("---")
        st.markdown("### 🛠️ Tools")
        selected_option = st.radio("Select Analysis Module", ["Image Analysis", "Audio Analysis", "Video Analysis"])
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user = None
            rerun()

    if selected_option == "Image Analysis":
        image_analysis_page()
    elif selected_option == "Audio Analysis":
        audio_analysis_page()
    elif selected_option == "Video Analysis":
        video_analysis_page()

def display_result(fake_prob, media_type, file_name, file_size, mime_type):
    real_prob = 1.0 - fake_prob
    confidence = max(fake_prob, real_prob) * 100
    is_fake = fake_prob > 0.5
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. Verdict Box
    css_class = "fake" if is_fake else "authentic"
    verdict_text = "⚠️ FAKE" if is_fake else "✅ AUTHENTIC MEDIA"
    
    st.markdown(f"""
    <div class="verdict-box {css_class}">
        <h1 style="color:white; margin:0; border:none;">{verdict_text}</h1>
        <h3 style="color:white; opacity:0.9; border:none; margin-top:10px;">Confidence Score: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Probability Meters
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌱 Real Probability")
        st.progress(max(0.0, min(1.0, real_prob)))
        st.markdown(f"**{real_prob*100:.2f}%**")
    with col2:
        st.markdown("### 🤖 Fake Probability")
        st.progress(max(0.0, min(1.0, fake_prob)))
        st.markdown(f"**{fake_prob*100:.2f}%**")

    # 3. Detailed Report
    st.markdown(f"""
    <div class="result-card">
        <h3>📊 Detailed Analysis Log</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #ddd;"><td style="padding:10px; font-weight:bold;">File Name</td><td style="padding:10px;">{file_name}</td></tr>
            <tr style="border-bottom: 1px solid #ddd;"><td style="padding:10px; font-weight:bold;">Media Type</td><td style="padding:10px;">{media_type}</td></tr>
            <tr style="border-bottom: 1px solid #ddd;"><td style="padding:10px; font-weight:bold;">MIME Type</td><td style="padding:10px;">{mime_type}</td></tr>
            <tr style="border-bottom: 1px solid #ddd;"><td style="padding:10px; font-weight:bold;">File Size</td><td style="padding:10px;">{format_size(file_size)}</td></tr>
            <tr style="border-bottom: 1px solid #ddd;"><td style="padding:10px; font-weight:bold;">Timestamp</td><td style="padding:10px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            <tr><td style="padding:10px; font-weight:bold;">Final Verdict</td><td style="padding:10px;">{'Fake' if is_fake else 'Real'}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

def image_analysis_page():
    st.markdown("## 🖼️ AI Media Authenticity Image Detector")
    st.info("Advanced ViT + FFT Analysis to detect Real and Fake Images.")
    
    uploaded_file = st.file_uploader("Drop your image here", type=['jpg', 'jpeg', 'png'])
    model, err, device = load_image_model()
    
    if err:
        st.error(f"System Error: {err}")
        return

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Source Image", use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ Analysis Console")
            run_analysis = st.button("🚀 Run Deep Analysis")

        if run_analysis:
            with st.spinner("🔍 Scanning pixel patterns..."):
                try:
                    rgb_np = np.array(image)
                    fft_np = fft_feature(rgb_np)
                    rgb_pil = Image.fromarray(rgb_np)
                    fft_pil = Image.fromarray(fft_np).convert("RGB")
                    
                    rgb_t = base_transform(rgb_pil)
                    fft_t = base_transform(fft_pil)
                    
                    x = torch.cat([rgb_t, fft_t], dim=0)
                    x = normalize_6ch(x)
                    x = x.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)
                        fake_prob = probs[0, 0].item()
                        
                    display_result(fake_prob, "IMAGE", uploaded_file.name, uploaded_file.size, uploaded_file.type)
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

def audio_analysis_page():
    st.markdown("## 🎙️ AI Media Authenticity Audio Detector")
    st.info("Spectral analysis to detect Real and Fake Audio")
    
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=['wav', 'mp3'])
    model, err = load_audio_model()
    
    if err:
        st.error(f"System Error: {err}")
        return
    
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("🚀 Analyze Audio"):
            with st.spinner("🎧 Listening for artifacts..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    mel_spec = preprocess_audio(tmp_path)
                    if mel_spec is not None:
                        # Reshape for model
                        X_new = mel_spec[np.newaxis, ..., np.newaxis]
                        prediction = model.predict(X_new, verbose=0)
                        fake_prob = float(prediction[0][0])
                        
                        display_result(fake_prob, "AUDIO", uploaded_file.name, uploaded_file.size, uploaded_file.type if uploaded_file.type else "audio/unknown")
                    else:
                        st.error("Audio Processing Error: No spectral data extracted. File might be silent or corrupt.")
                except Exception as e:
                    st.error(f"Audio Analysis Failed: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        try: os.unlink(tmp_path) 
                        except: pass

def video_analysis_page():
    st.markdown("## 🎥 AI Media Authenticity Video Detector")
    st.info("Frame-by-frame YOLO analysis to detect real and fake videos.")
    
    uploaded_file = st.file_uploader("Upload Video (MP4/AVI)", type=['mp4', 'avi', 'mov'])
    model, err = load_video_model()
    
    if err:
        st.error(f"System Error: {err}")
        return
    
    if uploaded_file:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.video(uploaded_file)
        
        with c2:
            st.markdown("### ⚙️ Action")
            run_video_scan = st.button("🚀 Analyze the video")
        
        if run_video_scan:
            with st.spinner("📽️ Scanning frames..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                tfile.close() # Important for Windows

                output_path = tempfile.mktemp(suffix='.mp4')
                
                cap = None
                out = None
                
                try:
                    cap = cv2.VideoCapture(video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    frame_count = 0
                    fake_detections = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        results = model(frame, conf=0.2, verbose=False)
                        annotated_frame = results[0].plot()
                        
                        # Only count detections where class is 0 (fake)
                        # Model classes: {0: 'fake', 1: 'real'}
                        if len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                if int(box.cls[0]) == 0:  # Class 0 = fake
                                    fake_detections += 1
                                    break  # Count frame as fake if any fake detection found
                        
                        out.write(annotated_frame)
                        frame_count += 1
                        
                    cap.release()
                    out.release()
                    cap = None
                    out = None
                    
                    # Calculate fake probability based on frames with fake detections
                    fake_prob = (fake_detections / frame_count) if frame_count > 0 else 0.0
                    
                    # Show Result
                    display_result(fake_prob, "VIDEO", uploaded_file.name, uploaded_file.size, "video/mp4")
                    
                    # (Optional) We could show the processed frames here, but user requested removal for cleaner UI
                        
                except Exception as e:
                    st.error(f"Video Processing Error: {e}")
                finally:
                    if cap and cap.isOpened(): cap.release()
                    if out: out.release()
                    
                    time.sleep(0.5)
                    if os.path.exists(video_path):
                        try: os.unlink(video_path)
                        except: pass
                    if os.path.exists(output_path):
                        try: os.unlink(output_path)
                        except: pass

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if st.session_state.authenticated:
        main_app()
    else:
        if st.session_state.page == "signup":
            signup_page()
        else:
            login_page()
