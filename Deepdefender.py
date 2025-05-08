import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
from torchvision import transforms
import yaml
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE


# Load config.yaml
def load_config():
    with open("model/config.yaml", "r") as f:
        return yaml.safe_load(f)

# def load_models():
#     config = load_config()

#     # Load ED
#     model_ed = GenConViTED(config=config, pretrained=True)
#     checkpoint_ed = torch.load("weight/GenConViT_ED.pth", map_location="cpu")
#     state_ed = model_ed.state_dict()
#     matched_ed = {k: v for k, v in checkpoint_ed.items() if k in state_ed and state_ed[k].shape == v.shape}
#     state_ed.update(matched_ed)
#     model_ed.load_state_dict(state_ed, strict=False)
#     model_ed.eval()

#     # Load VAE
#     model_vae = GenConViTVAE(config=config, pretrained=True)
#     checkpoint_vae = torch.load("weight/GenConViT_VAE.pth", map_location="cpu")
#     state_vae = model_vae.state_dict()
#     matched_vae = {k: v for k, v in checkpoint_vae.items() if k in state_vae and state_vae[k].shape == v.shape}
#     state_vae.update(matched_vae)
#     model_vae.load_state_dict(state_vae, strict=False)
#     model_vae.eval()

#     return model_ed, model_vae

def load_models():
    config = load_config()

    # Load ED
    model_ed = GenConViTED(config=config, pretrained=True)
    checkpoint_ed = torch.load("weight/best_model.pt", map_location="cpu")  # Changed to best_model.pt
    state_ed = model_ed.state_dict()
    matched_ed = {k: v for k, v in checkpoint_ed.items() if k in state_ed and state_ed[k].shape == v.shape}
    state_ed.update(matched_ed)
    model_ed.load_state_dict(state_ed, strict=False)
    model_ed.eval()

    # If you have a single model file instead of two separate ones
    model_vae = GenConViTVAE(config=config, pretrained=True)
    checkpoint_vae = torch.load("weight/best_model.pt", map_location="cpu")  # Same file or another .pt file
    state_vae = model_vae.state_dict()
    matched_vae = {k: v for k, v in checkpoint_vae.items() if k in state_vae and state_vae[k].shape == v.shape}
    state_vae.update(matched_vae)
    model_vae.load_state_dict(state_vae, strict=False)
    model_vae.eval()

    return model_ed, model_vae

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function for prediction
def predict_frame(model_ed, model_vae, image_tensor):
    with torch.no_grad():
        # Run ED model
        logits_ed = model_ed(image_tensor)
        
        # Run VAE model (returns logits and reconstructed image)
        logits_vae, _ = model_vae(image_tensor)
        
        # Average their logits for ensemble prediction
        combined_logits = (logits_ed + logits_vae) / 2
        
        # Get softmax probabilities and prediction
        probs = torch.softmax(combined_logits, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_idx].item()
        
        return pred_class_idx, confidence

# Streamlit UI
st.title("🎥 GenConViT Deepfake Detector")
st.write("Upload a video to detect deepfakes using both model variants.")

# Load models on app startup
try:
    model_ed, model_vae = load_models()
    st.success("Models loaded successfully! Ready to analyze videos.")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(uploaded_file)

    # Open video and get metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file. Please try another file.")
        os.unlink(video_path)
        st.stop()
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    st.write(f"📸 Total frames: {total_frames}")
    st.write(f"⏱ Video duration: {duration:.2f} seconds")

    # Determine sampling rate (adjust based on video length)
    sampling_rate = max(1, int(total_frames / 30))  # Sample ~30 frames max
    st.write(f"⏱ Sampling every {sampling_rate} frames...")

    frame_preds = []
    frame_count = 0
    progress_bar = st.progress(0)

    with st.spinner("Analyzing video..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sampling_rate == 0:
                try:
                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    input_tensor = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
                    
                    # Get prediction
                    pred_class_idx, confidence = predict_frame(model_ed, model_vae, input_tensor)
                    frame_preds.append((pred_class_idx, confidence))
                    
                except Exception as e:
                    st.warning(f"Error processing frame {frame_count}: {str(e)}")
                    continue

            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        cap.release()

    # Count predictions
    real_count = sum(1 for p, _ in frame_preds if p == 0)
    fake_count = sum(1 for p, _ in frame_preds if p == 1)
    
    # Determine final prediction
    if frame_preds:
        # Use majority voting
        final_pred = 1 if fake_count > real_count else 0
        label = "Fake" if final_pred == 1 else "Real"
        
        # Calculate confidence
        fake_ratio = fake_count / (real_count + fake_count)
        confidence = max(fake_ratio, 1 - fake_ratio)  # How confident in the majority class
        
        # Show results
        result_color = "#FF4B4B" if final_pred == 1 else "#4BB543"
        st.markdown(
            f"<h2 style='color:{result_color};'>Final Prediction: {label} ({confidence*100:.1f}% confidence)</h2>", 
            unsafe_allow_html=True
        )
        
        # Show frame counts
        st.write("### Frame-wise Analysis:")
        st.write(f"✅ Real frames detected: {real_count} ({real_count/(real_count+fake_count)*100:.1f}%)")
        st.write(f"❌ Fake frames detected: {fake_count} ({fake_count/(real_count+fake_count)*100:.1f}%)")
        
        # Average confidence of individual frame predictions
        avg_conf = np.mean([conf for _, conf in frame_preds])
        st.write(f"🔍 Average frame prediction confidence: **{avg_conf * 100:.2f}%**")
        
    else:
        st.error("⚠️ No frames were processed. Try another video.")

    # Clean up temporary file
    os.unlink(video_path)
else:
    st.info("Please upload a video file to analyze")