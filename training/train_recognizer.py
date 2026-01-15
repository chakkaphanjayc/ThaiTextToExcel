from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.ocr_pipeline import (
    CRNN,
    OcrDataset,
    build_vocab,
    collate_fn,
    compute_cer,
    compute_wer,
    greedy_decode,
    load_labels,
    save_vocab,
)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, vocab: dict) -> tuple[float, float]:
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
    return cer, wer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCR recognizer")
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_labels = load_labels(args.train_labels)
    val_labels = load_labels(args.val_labels)

    vocab = build_vocab([text for _, text in train_labels])
    save_vocab(vocab, args.output_dir / "vocab.json")

    train_dataset = OcrDataset(train_labels, args.train_images, vocab)
    val_dataset = OcrDataset(val_labels, args.val_images, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    num_classes = len(vocab) + 1
    model = CRNN(num_classes)
    device = torch.device(args.device)
    model.to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_cer = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch.images.to(device)
            targets = batch.targets.to(device)
            target_lengths = batch.target_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2)
            input_lengths = torch.full(
                (logits.size(1),),
                logits.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        cer, wer = evaluate(model, val_loader, device, vocab)
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}: loss={avg_loss:.4f} CER={cer:.4f} WER={wer:.4f}")

        if cer < best_cer:
            best_cer = cer
            torch.save(model.state_dict(), args.output_dir / "ocr_crnn.pt")
            print("Saved best model")


if __name__ == "__main__":
    main()
