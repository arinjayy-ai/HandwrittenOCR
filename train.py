import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import get_datasets, collate_fn
from model import CRNN


def train():

    print("=" * 60)
    print("Device :", config.DEVICE)
    print("=" * 60)

    # ----------------------------------------------------
    # Dataset
    # ----------------------------------------------------

    train_dataset, val_dataset, test_dataset, vocab = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    print(f"\nVocabulary Size : {len(vocab)}")

    # ----------------------------------------------------
    # Model
    # ----------------------------------------------------

    model = CRNN(len(vocab)).to(config.DEVICE)

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    # ----------------------------------------------------
    # Resume checkpoint
    # ----------------------------------------------------

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if os.path.exists(config.LAST_MODEL_PATH):

        checkpoint = torch.load(
            config.LAST_MODEL_PATH,
            map_location=config.DEVICE
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]

        print(f"\nResuming from Epoch {start_epoch}")

    # ----------------------------------------------------
    # Training Loop
    # ----------------------------------------------------

    for epoch in range(start_epoch, config.NUM_EPOCHS):

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        model.train()

        train_loss = 0

        progress = tqdm(train_loader)

        for images, labels, label_lengths in progress:

            images = images.to(config.DEVICE)

            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(config.DEVICE)

                log_probs = outputs.log_softmax(2)

                loss = criterion(
                    log_probs,
                    labels,
                    input_lengths,
                    label_lengths
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= len(train_loader)

        # ==================================================
        # Validation
        # ==================================================

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for images, labels, label_lengths in val_loader:

                images = images.to(config.DEVICE)

                labels = labels.to(config.DEVICE)

                outputs = model(images)

                input_lengths = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(config.DEVICE)

                log_probs = outputs.log_softmax(2)

                loss = criterion(
                    log_probs,
                    labels,
                    input_lengths,
                    label_lengths
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print()

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Validation Loss : {val_loss:.4f}")

        # ==================================================
        # Save Last Model
        # ==================================================

        torch.save({

            "epoch": epoch,

            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "best_loss": best_val_loss

        }, config.LAST_MODEL_PATH)

        # ==================================================
        # Save Best Model
        # ==================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save({

                "epoch": epoch,

                "model": model.state_dict(),

                "optimizer": optimizer.state_dict(),

                "best_loss": best_val_loss

            }, config.BEST_MODEL_PATH)

            print("Best Model Saved")

            patience_counter = 0

        else:

            patience_counter += 1

            print(f"No Improvement ({patience_counter})")

        # ==================================================
        # Early Stopping
        # ==================================================

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:

            print("\nEarly Stopping Triggered")

            break

    print("\nTraining Complete")


if __name__ == "__main__":
    train()