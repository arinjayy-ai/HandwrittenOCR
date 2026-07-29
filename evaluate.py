import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jiwer import cer, wer
from tqdm import tqdm

import config
from dataset import get_datasets, collate_fn
from model import CRNN


# ==========================================================
# Greedy CTC Decoder
# ==========================================================

def greedy_decode(output, vocab):

    output = output.softmax(2)

    prediction = output.argmax(2)

    prediction = prediction.permute(1, 0)

    predictions = []

    for sequence in prediction:

        text = vocab.decode(sequence.cpu().numpy())

        predictions.append(text)

    return predictions


# ==========================================================
# Evaluate
# ==========================================================

def evaluate():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    _, _, test_dataset, vocab = get_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    print("\nLoading Model...")

    model = CRNN(len(vocab))

    checkpoint = torch.load(
        config.BEST_MODEL_PATH,
        map_location=config.DEVICE
    )

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)

    model.eval()

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    total_loss = 0

    ground_truths = []

    predictions = []

    csv_rows = []

    print("\nRunning Evaluation...\n")

    with torch.no_grad():

        for images, labels, label_lengths in tqdm(test_loader):

            images = images.to(config.DEVICE)

            labels = labels.to(config.DEVICE)

            outputs = model(images)

            input_lengths = torch.full(
                (outputs.size(1),),
                outputs.size(0),
                dtype=torch.long
            ).to(config.DEVICE)

            log_probs = outputs.log_softmax(2)

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item()

            batch_predictions = greedy_decode(outputs, vocab)

            start = 0

            for i, length in enumerate(label_lengths):

                target = labels[start:start+length]

                target_text = vocab.decode(target.cpu().numpy())

                pred_text = batch_predictions[i]

                ground_truths.append(target_text)
                predictions.append(pred_text)

                csv_rows.append([
                    target_text,
                    pred_text
                ])

                start += length

    test_loss = total_loss / len(test_loader)

    # ======================================================
    # Metrics
    # ======================================================

    cer_score = cer(ground_truths, predictions)

    wer_score = wer(ground_truths, predictions)

    word_accuracy = (
        sum(
            gt == pred
            for gt, pred in zip(
                ground_truths,
                predictions
            )
        )
        / len(ground_truths)
    ) * 100

    character_accuracy = (1 - cer_score) * 100

    # ======================================================
    # Save CSV
    # ======================================================

    output_csv = os.path.join(
        config.OUTPUT_DIR,
        "predictions.csv"
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Ground Truth",
            "Prediction"
        ])

        writer.writerows(csv_rows)

    # ======================================================
    # Print Results
    # ======================================================

    print("\n")
    print("=" * 60)

    print(f"Test Loss           : {test_loss:.4f}")

    print(f"Character Accuracy  : {character_accuracy:.2f}%")

    print(f"Word Accuracy       : {word_accuracy:.2f}%")

    print(f"CER                : {cer_score:.4f}")

    print(f"WER                : {wer_score:.4f}")

    print("=" * 60)

    print(f"\nPredictions saved to:\n{output_csv}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    evaluate()