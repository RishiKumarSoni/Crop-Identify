# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showImageFormat', False)


# Cache model load
@st.cache_resource
def load_model(model_path="models/best_10cls_982.pt"):
    from ultralytics import YOLO
    model = YOLO(model_path)
    class_names = list(model.names.values())
    return model, class_names

def load_css(path="static/style.css"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def predict(model, pil_img):
    arr = np.array(pil_img.convert("RGB"))
    # w, h = arr.shape[:2]
    if arr.shape[0]>1024:
        shape = (640, 640)
    else:
        shape = (224, 224)
    results = model.predict(source=arr, task="classify", imgsz=shape, verbose=False)
    r = results[0]

    # Extract probabilities
    probs = r.probs.data.cpu().numpy().ravel()
    return probs

def main():
    st.set_page_config(page_title="Crop Identification", layout="centered")
    load_css("static/style.css")

    st.markdown('<div class="app-container">', unsafe_allow_html=True)
    st.markdown('<div class="app-header"><h2>Crop Identification</h2></div>', unsafe_allow_html=True)
    

    # Sidebar logo only
    st.sidebar.image("assets/annam_logo.png", width="stretch")  
    # Optional: still keep model path hidden in backend
    model_path = "models/best_10cls_982.pt"

    # Load YOLO model
    try:
        model, class_names = load_model(model_path)
        # st.sidebar.success(f"Loaded {len(class_names)} classes")
        st.sidebar.success(f"Welcome!")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return

    # --- Initialize session state ---
    if "input_type" not in st.session_state:
        st.session_state.input_type = None
    if "pil_img" not in st.session_state:
        st.session_state.pil_img = None
    # Ensure probs exists in session_state
    if "probs" not in st.session_state:
        st.session_state.probs = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    col1, col2 = st.columns([5, 1])

    with col2:
        if st.button("🔄 Refresh"):
            for key in ["pil_img", "input_type", "probs", "show_camera"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Increment uploader key to reset the widget
            st.session_state.uploader_key += 1
            
            st.rerun()

    # --- Upload option ---
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state.input_type = "upload"
        st.session_state.pil_img = Image.open(uploaded).convert("RGB")

     # --- Camera option ---
    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False

    # Create two columns

    with col1:
        if st.button("Take a photo"):
            st.session_state.show_camera = True
            st.session_state.input_type = "camera"
            st.session_state.pil_img = None   # reset previous

    camera_img = None
    if st.session_state.show_camera:
        placeholder = st.empty()
        temp_img = placeholder.camera_input("Camera")
        if temp_img is not None:
            st.session_state.pil_img = Image.open(temp_img).convert("RGB")
            st.session_state.show_camera = False
            placeholder.empty()

    # --- Display final image ---
    if st.session_state.pil_img is not None:
        if st.session_state.input_type == "upload":
            st.image(st.session_state.pil_img, caption="Input image (from upload)")
        elif st.session_state.input_type == "camera":
            st.image(st.session_state.pil_img, caption="Input image (from camera)")
    else:
        st.info("Upload an image or take a photo to identify.")


    # Predict
    if st.session_state.pil_img is not None:
        with st.spinner("Running inference..."):
            try:
                st.session_state.probs = predict(model, st.session_state.pil_img)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

    # Normalize
    # probs = probs / probs.sum()
    if st.session_state.get("probs") is not None:
        # st.session_state.probs = st.session_state.probs / st.session_state.probs.sum()
        probs = st.session_state.probs
        if not np.isclose(probs.sum(), 1.0):
            st.session_state.probs = probs / probs.sum()

        # Top prediction
        top_idx = int(np.argmax(probs))
        top_name = class_names[top_idx]
        top_conf = float(probs[top_idx])

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### Prediction: **{top_name}** — {top_conf*100:.2f}%")
        st.write("Top probabilities:")

        # Show top N
        pairs = sorted(zip(class_names, probs), key=lambda x: -x[1])
        for nm, p in pairs[:10]:
            st.markdown(f"- **{nm}** — {p*100:.2f}%")

        # Bar chart
        try:
            import pandas as pd
            df = pd.DataFrame({"class": [p[0] for p in pairs[:20]], "prob": [p[1] for p in pairs[:20]]})
            st.bar_chart(df.set_index("class"))
        except Exception:
            pass

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
