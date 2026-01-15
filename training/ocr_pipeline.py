from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn


@dataclass
class Batch:
    images: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    references: List[str]


def load_labels(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if filename:
                rows.append((filename, text))
        return rows


def build_vocab(texts: Iterable[str]) -> Dict[str, int]:
    chars = sorted({char for text in texts for char in text})
    return {char: idx + 1 for idx, char in enumerate(chars)}


def save_vocab(vocab: Dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(k): int(v) for k, v in data.items()}


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {index: char for char, index in vocab.items()}


def encode_text(text: str, vocab: Dict[str, int]) -> torch.Tensor:
    indices = [vocab[char] for char in text if char in vocab]
    return torch.tensor(indices, dtype=torch.long)


def resize_image(image: Image.Image, height: int = 32) -> Image.Image:
    width, current_height = image.size
    if current_height == 0:
        return image
    new_width = max(1, int(width * (height / current_height)))
    return image.resize((new_width, height), Image.BILINEAR)


class OcrDataset(torch.utils.data.Dataset):
    def __init__(self, labels: List[Tuple[str, str]], images_dir: Path, vocab: Dict[str, int]):
        self.labels = labels
        self.images_dir = images_dir
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filename, text = self.labels[index]
        image = Image.open(self.images_dir / filename).convert("L")
        image = resize_image(image)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        label_tensor = encode_text(text, self.vocab)
        return image_tensor, label_tensor, text


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]) -> Batch:
    images, labels, references = zip(*batch)
    heights = [image.shape[1] for image in images]
    widths = [image.shape[2] for image in images]
    max_height = max(heights)
    max_width = max(widths)

    padded_images = torch.zeros(len(images), 1, max_height, max_width)
    for i, image in enumerate(images):
        padded_images[i, :, : image.shape[1], : image.shape[2]] = image

    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    targets = torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)

    return Batch(images=padded_images, targets=targets, target_lengths=label_lengths, references=list(references))


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)
        recurrent, _ = self.rnn(features)
        return self.fc(recurrent)


def greedy_decode(logits: torch.Tensor, vocab: Dict[str, int]) -> List[str]:
    inverse_vocab = invert_vocab(vocab)
    probs = logits.permute(1, 0, 2)
    predictions = probs.argmax(dim=2)

    texts = []
    for sequence in predictions:
        chars = []
        prev = None
        for index in sequence.tolist():
            if index != 0 and index != prev:
                chars.append(inverse_vocab.get(index, ""))
            prev = index
        texts.append("".join(chars))
    return texts


def edit_distance(a: Iterable[str], b: Iterable[str]) -> int:
    a_list = list(a)
    b_list = list(b)
    dp = np.zeros((len(a_list) + 1, len(b_list) + 1), dtype=int)
    for i in range(len(a_list) + 1):
        dp[i][0] = i
    for j in range(len(b_list) + 1):
        dp[0][j] = j
    for i in range(1, len(a_list) + 1):
        for j in range(1, len(b_list) + 1):
            cost = 0 if a_list[i - 1] == b_list[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[len(a_list)][len(b_list)])


def compute_cer(predictions: List[str], references: List[str]) -> float:
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_distance += edit_distance(list(pred), list(ref))
        total_chars += len(ref)
    return total_distance / max(1, total_chars)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    total_distance = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_distance += edit_distance(pred_words, ref_words)
        total_words += len(ref_words)
    return total_distance / max(1, total_words)
