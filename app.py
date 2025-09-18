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

    # Sidebar: model
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

    # Upload image
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # camera input option
    camera_img = st.camera_input("Take a photo") 

    # Decide which input to use
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Input image (from upload)", width="stretch", output_format="JPEG")

    elif camera_img is not None:
        pil_img = Image.open(camera_img).convert("RGB")
        st.image(pil_img, caption="Input image (from camera)", width="stretch", output_format="JPEG")

    else:
        st.info("Upload an image or take a photo to classify.")
        st.markdown('</div>', unsafe_allow_html=True)
        return


    # Predict
    with st.spinner("Running inference..."):
        try:
            probs = predict(model, pil_img)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    # Normalize
    probs = probs / probs.sum()

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
