from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.ocr_pipeline import (
    CRNN,
    OcrDataset,
    collate_fn,
    compute_cer,
    compute_wer,
    greedy_decode,
    load_labels,
    load_vocab,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR recognizer")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    vocab = load_vocab(args.vocab_path)

    dataset = OcrDataset(labels, args.images_dir, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = CRNN(len(vocab) + 1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    references = []
    with torch.no_grad():
        for batch in loader:
            images = batch.images.to(device)
            logits = model(images)
            decoded = greedy_decode(logits.cpu(), vocab)
            predictions.extend(decoded)
            references.extend(batch.references)

    cer = compute_cer(predictions, references)
    wer = compute_wer(predictions, references)
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")


if __name__ == "__main__":
    main()
