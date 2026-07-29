import os
import argparse

import torch
from PIL import Image
from torchvision import transforms

import config
from model import CRNN
from vocabulary import Vocabulary


# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ==========================================================
# Load Vocabulary
# ==========================================================

vocab = Vocabulary()

vocab.load(
    os.path.join(
        config.CHECKPOINT_DIR,
        "vocabulary.pkl"
    )
)


# ==========================================================
# Load Model
# ==========================================================

model = CRNN(len(vocab))

checkpoint = torch.load(
    config.BEST_MODEL_PATH,
    map_location=config.DEVICE
)

# Compatible with both state_dict and checkpoint dictionary
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.to(config.DEVICE)

model.eval()


# ==========================================================
# CTC Greedy Decoder
# ==========================================================

def greedy_decode(output):

    output = output.softmax(2)

    prediction = output.argmax(2)

    prediction = prediction.squeeze(1).cpu().numpy()

    text = vocab.decode(prediction)

    return text


# ==========================================================
# Predict Function
# ==========================================================

def predict(image_path):

    image = Image.open(image_path).convert("L")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(config.DEVICE)

    with torch.no_grad():

        output = model(image)

    text = greedy_decode(output)

    return text


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to image"
    )

    args = parser.parse_args()

    print("=" * 50)

    print("Image :", args.image)

    prediction = predict(args.image)

    print("Prediction :", prediction)

    print("=" * 50)