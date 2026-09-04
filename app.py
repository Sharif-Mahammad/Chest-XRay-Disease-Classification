import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from train_compare import build_model


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chest X-Ray Disease Detection",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

MODEL_PATH = "outputs/efficientnet_b0_best.pt"

DISPLAY_NAME_MAP = {
    "COVID19": "COVID-19",
    "PNEUMONIA": "Pneumonia",
    "TURBERCULOSIS": "Tuberculosis"
}


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    model = build_model(
        "efficientnet_b0",
        num_classes=len(class_names),
        pretrained=False
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, class_names, img_size


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image, img_size):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image).unsqueeze(0)

    return tensor


# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------

def generate_gradcam(model, image, img_size):

    tensor = preprocess_image(image, img_size)

    target_layer = model.features[-1]

    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    # Forward pass
    output = model(tensor)

    probabilities = F.softmax(output, dim=1)

    predicted_index = int(
        output.argmax(dim=1).item()
    )

    confidence = float(
        probabilities[0, predicted_index].item()
    )

    # Backward pass
    model.zero_grad()

    output[0, predicted_index].backward()

    grads = gradients["value"]
    acts = activations["value"]

    # Calculate Grad-CAM weights
    weights = grads.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (weights * acts).sum(
        dim=1,
        keepdim=True
    )

    cam = F.interpolate(
        cam,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

    # ReLU
    cam = np.maximum(cam, 0)

    # Normalize
    cam = cam / (cam.max() + 1e-8)

    # Resize original image
    original_image = image.resize(
        (img_size, img_size)
    )

    # Create heatmap
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[:, :, :3]

    # Convert original image to numpy
    original_array = np.array(
        original_image
    ) / 255.0

    # Overlay heatmap
    overlay = (
        0.6 * original_array
        + 0.4 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return (
        predicted_index,
        confidence,
        probabilities[0].detach().numpy(),
        overlay
    )


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🩺 Chest X-Ray Disease Classification")

st.markdown(
    """
    ### AI-powered chest X-ray classification using EfficientNet-B0

    Upload a chest X-ray image to obtain a predicted condition
    category along with a Grad-CAM visualization showing the
    regions that influenced the model's prediction.
    """
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** EfficientNet-B0")
    st.write("**Task:** 3-Class Classification")
    st.write("**Classes:**")
    st.write("- COVID-19")
    st.write("- Pneumonia")
    st.write("- Tuberculosis")

    st.divider()

    st.warning(
        """
        **Medical Disclaimer**

        This application is developed for educational
        and research purposes only.

        It is not a medical diagnostic tool and should
        not be used as a substitute for professional
        medical advice.
        """
    )


# --------------------------------------------------
# Load Model
# --------------------------------------------------

try:

    model, class_names, img_size = load_model()

except Exception as e:

    st.error(
        f"Unable to load the trained model: {e}"
    )

    st.stop()


# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Analysis
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("📷 Uploaded X-Ray")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width=500
    )

    if st.button(
        "🔍 Analyze X-Ray",
        type="primary"
    ):

        with st.spinner(
            "Analyzing X-ray..."
        ):

            (
                predicted_index,
                confidence,
                probabilities,
                gradcam_image
            ) = generate_gradcam(
                model,
                image,
                img_size
            )

        predicted_class = class_names[
            predicted_index
        ]

        display_name = DISPLAY_NAME_MAP.get(
            predicted_class,
            predicted_class
        )

        # ------------------------------------------
        # Prediction Result
        # ------------------------------------------

        st.divider()

        st.subheader("🎯 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Condition",
                display_name
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        # ------------------------------------------
        # Class Probabilities
        # ------------------------------------------

        st.subheader(
            "📊 Class Probabilities"
        )

        for index, class_name in enumerate(
            class_names
        ):

            display_class = DISPLAY_NAME_MAP.get(
                class_name,
                class_name
            )

            probability = probabilities[index]

            st.write(
                f"**{display_class}**"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                f"{probability * 100:.2f}%"
            )


        # ------------------------------------------
        # Grad-CAM
        # ------------------------------------------

        st.divider()

        st.subheader(
            "🔥 Grad-CAM Explainability"
        )

        st.write(
            """
            The highlighted regions indicate areas of the
            X-ray that contributed more strongly to the model's
            prediction.
            """
        )

        st.image(
            gradcam_image,
            caption=f"Grad-CAM — Predicted: {display_name}",
            width=600
        )

else:

    st.info(
        "👆 This model performs single-class classification and predicts "
        "the most likely condition among COVID-19, Pneumonia, and Tuberculosis."
    )