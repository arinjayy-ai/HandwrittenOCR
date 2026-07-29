import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

import config
from model import CRNN
from vocabulary import Vocabulary


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Handwritten Text Recognition")

st.write(
    "Upload a handwritten word image and the model will recognise the text."
)


# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(
        (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --------------------------------------------------
# Load Vocabulary
# --------------------------------------------------

vocab = Vocabulary()

vocab.load(
    os.path.join(
        config.CHECKPOINT_DIR,
        "vocabulary.pkl"
    )
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = CRNN(len(vocab))

    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE
    )

    if "model" in checkpoint:
        model.load_state_dict(
            checkpoint["model"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)

    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# Decoder
# --------------------------------------------------

def decode(output):

    prediction = output.softmax(2)

    prediction = prediction.argmax(2)

    prediction = prediction.squeeze(1)

    return vocab.decode(
        prediction.cpu().numpy()
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.convert("L")

    img = transform(img)

    img = img.unsqueeze(0)

    img = img.to(config.DEVICE)

    with torch.no_grad():

        output = model(img)

    prediction = decode(output)

    st.success(
        f"Predicted Text: **{prediction}**"
    )