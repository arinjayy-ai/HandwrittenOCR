import os
from PIL import Image

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config
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
# Load IAM Annotation File
# ==========================================================

def load_annotations():

    images = []
    texts = []

    with open(config.ANNOTATION_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 9:
                continue

            image_id = parts[0]

            status = parts[1]

            if status != "ok":
                continue

            word = " ".join(parts[8:])

            ids = image_id.split("-")

            folder1 = ids[0]
            folder2 = ids[0] + "-" + ids[1]

            image_path = os.path.join(
                config.IMAGE_FOLDER,
                folder1,
                folder2,
                image_id + ".png"
            )

            if not os.path.exists(image_path):
                continue

            try:
                Image.open(image_path).verify()
            except:
                print("Corrupted image skipped:", image_path)
                continue

            images.append(image_path)
            texts.append(word)

    df = pd.DataFrame({
        "image_path": images,
        "text": texts
    })

    print(f"Loaded {len(df)} valid samples")

    return df
# ==========================================================
# OCR Dataset
# ==========================================================

class OCRDataset(Dataset):

    def __init__(self, dataframe, vocabulary):

        self.df = dataframe.reset_index(drop=True)

        self.vocab = vocabulary

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(row["image_path"]).convert("L")
        image = transform(image)

        text = row["text"]

        label = self.vocab.encode(text)

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label_tensor
    # ==========================================================
# Collate Function
# ==========================================================

def collate_fn(batch):

    images = []
    labels = []
    label_lengths = []

    for image, label in batch:

        images.append(image)

        labels.extend(label.tolist())

        label_lengths.append(len(label))

    images = torch.stack(images)

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    label_lengths = torch.tensor(
        label_lengths,
        dtype=torch.long
    )

    return images, labels, label_lengths

# ==========================================================
# Dataset Builder
# ==========================================================

def get_datasets():

    df = load_annotations()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=config.SEED,
        shuffle=True
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=config.SEED,
        shuffle=True
    )

    print()

    print("Training :", len(train_df))
    print("Validation :", len(val_df))
    print("Testing :", len(test_df))

    vocab = Vocabulary()

    vocab.build(train_df["text"].tolist())

    vocab.save(
        os.path.join(
            config.CHECKPOINT_DIR,
            "vocabulary.pkl"
        )
    )

    train_dataset = OCRDataset(train_df, vocab)
    val_dataset = OCRDataset(val_df, vocab)
    test_dataset = OCRDataset(test_df, vocab)

    return train_dataset, val_dataset, test_dataset, vocab
if __name__ == "__main__":

    train_ds, val_ds, test_ds, vocab = get_datasets()

    print()

    print("Vocabulary Size :", len(vocab))

    print("Train :", len(train_ds))
    print("Validation :", len(val_ds))
    print("Test :", len(test_ds))

    image, label = train_ds[0]

    print()

    print("Image Shape :", image.shape)

    print("Encoded Label :", label)

    print("Decoded :", vocab.decode(label.tolist()))