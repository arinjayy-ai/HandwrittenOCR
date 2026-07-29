"""
vocabulary.py

Creates and manages the OCR vocabulary.

CTC Blank Token = 0
"""

import os
import pickle

# ==========================================================
# Vocabulary Class
# ==========================================================

class Vocabulary:

    def __init__(self):

        # CTC blank token
        self.blank_token = "<BLANK>"

        self.char_to_idx = {
            self.blank_token: 0
        }

        self.idx_to_char = {
            0: self.blank_token
        }

    # ======================================================
    # Build vocabulary
    # ======================================================

    def build(self, texts):

        characters = set()

        for text in texts:
            if text is None:
                continue

            for ch in text:
                characters.add(ch)

        characters = sorted(list(characters))

        index = 1

        for ch in characters:

            self.char_to_idx[ch] = index
            self.idx_to_char[index] = ch

            index += 1

        print("=" * 50)
        print("Vocabulary Built Successfully")
        print(f"Characters : {len(characters)}")
        print(f"Vocabulary Size : {len(self.char_to_idx)}")
        print("=" * 50)

    # ======================================================
    # Encode text
    # ======================================================

    def encode(self, text):

        return [
            self.char_to_idx[ch]
            for ch in text
            if ch in self.char_to_idx
        ]

    # ======================================================
    # Decode prediction
    # ======================================================

    def decode(self, indices):

        result = []

        previous = None

        for idx in indices:

            idx = int(idx)

            # Ignore blanks
            if idx == 0:
                previous = idx
                continue

            # Remove repeated characters
            if idx == previous:
                continue

            result.append(
                self.idx_to_char[idx]
            )

            previous = idx

        return "".join(result)

    # ======================================================
    # Save vocabulary
    # ======================================================

    def save(self, path):

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "char_to_idx": self.char_to_idx,
                    "idx_to_char": self.idx_to_char
                },
                f
            )

        print(f"Vocabulary saved to {path}")

    # ======================================================
    # Load vocabulary
    # ======================================================

    def load(self, path):

        with open(path, "rb") as f:

            data = pickle.load(f)

        self.char_to_idx = data["char_to_idx"]
        self.idx_to_char = data["idx_to_char"]

        print(f"Vocabulary loaded from {path}")

    # ======================================================
    # Length
    # ======================================================

    def __len__(self):

        return len(self.char_to_idx)