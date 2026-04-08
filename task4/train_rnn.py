import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data import load_imdb_data
from src.evaluate import (
    classification_metrics,
    compute_confusion_matrix,
    save_confusion_matrix_figure,
)
from src.utils import save_json

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Embedding + LSTM on IMDB.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--metrics-path", type=str, default="outputs/metrics/rnn_metrics.json")
    parser.add_argument(
        "--cm-fig-path",
        type=str,
        default="outputs/figures/rnn_confusion_matrix.png",
    )
    parser.add_argument(
        "--curve-fig-path",
        type=str,
        default="outputs/figures/rnn_training_curve.png",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/model/rnn_state_dict.pt",
    )
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    # RNN input uses token sequences directly, not TF-IDF vectors.
    return text.split()


def build_vocab(train_texts: List[str], max_vocab_size: int) -> Dict[str, int]:
    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    most_common_tokens = counter.most_common(max_vocab_size - len(vocab))
    for token, _ in most_common_tokens:
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> Tuple[List[int], int]:
    token_ids = [vocab.get(token, UNK_ID) for token in tokenize(text)]
    token_ids = token_ids[:max_length]
    length = len(token_ids)
    if length < max_length:
        token_ids.extend([PAD_ID] * (max_length - length))
    return token_ids, max(1, length)


class IMDBSequenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int):
        self.labels = labels
        self.sequences = []
        self.lengths = []
        for text in texts:
            seq, length = encode_text(text, vocab, max_length)
            self.sequences.append(seq)
            self.lengths.append(length)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        return sequence, label, length


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)
        batch_size = outputs.size(0)
        idx = (lengths - 1).clamp(min=0)
        last_outputs = outputs[torch.arange(batch_size, device=outputs.device), idx]
        logits = self.fc(last_outputs).squeeze(1)
        return logits


@torch.no_grad()
def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for sequences, labels, lengths in data_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(sequences, lengths)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.long().cpu().tolist())
    return all_labels, all_preds


def save_training_curve(history: Dict[str, List[float]], save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], marker="o")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs, history["test_accuracy"], marker="o", label="Accuracy")
    ax2.plot(epochs, history["test_f1"], marker="o", label="F1")
    ax2.set_title("Test Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on GPU: {device.type == 'cuda'}")
    print("RNN input is token-id sequences (not TF-IDF vectors).")

    # 1) Load IMDB train/test split (same split source as TF-IDF baselines)
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(lowercase=True)

    # 2) Build vocabulary from training texts only, then encode as fixed-length sequences
    vocab = build_vocab(train_texts, max_vocab_size=args.max_vocab_size)
    train_dataset = IMDBSequenceDataset(train_texts, train_labels, vocab, max_length=args.max_length)
    test_dataset = IMDBSequenceDataset(test_texts, test_labels, vocab, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3) Build and train LSTM classifier
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "test_accuracy": [], "test_f1": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(sequences, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        eval_labels, eval_preds = evaluate_model(model, test_loader, device)
        eval_metrics = classification_metrics(eval_labels, eval_preds)

        history["train_loss"].append(float(avg_train_loss))
        history["test_accuracy"].append(eval_metrics["accuracy"])
        history["test_f1"].append(eval_metrics["f1-score"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"test_acc={eval_metrics['accuracy']:.4f} | "
            f"test_f1={eval_metrics['f1-score']:.4f}"
        )

    # 4) Final evaluation and outputs
    y_true, y_pred = evaluate_model(model, test_loader, device)
    metrics = classification_metrics(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)

    print("Final Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("Confusion Matrix:")
    print(cm)

    metrics_to_save = {
        **metrics,
        "confusion_matrix": cm,
        "train_loss": history["train_loss"],
        "test_accuracy_by_epoch": history["test_accuracy"],
        "test_f1_by_epoch": history["test_f1"],
        "model_config": {
            "max_length": args.max_length,
            "vocab_size": len(vocab),
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": str(device),
            "used_gpu": device.type == "cuda",
        },
    }

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_path)

    save_json(metrics_to_save, args.metrics_path)
    save_confusion_matrix_figure(cm, args.cm_fig_path)
    save_training_curve(history, args.curve_fig_path)

    print(f"\nMetrics saved to: {args.metrics_path}")
    print(f"Confusion matrix figure saved to: {args.cm_fig_path}")
    print(f"Training curve figure saved to: {args.curve_fig_path}")
    print(f"Model state_dict saved to: {args.model_path}")


if __name__ == "__main__":
    main()
