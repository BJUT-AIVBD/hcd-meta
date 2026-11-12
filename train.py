import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataloader import ConViS, HCDMeta, make_loader
from model import HCVR


def class_batches(dataset, rng):
    groups = {}
    for index, record in enumerate(dataset.records):
        groups.setdefault(record["keyword"], []).append(index)
    groups = [indices for indices in groups.values() if len(indices) >= 2]
    if len(groups) < 2:
        raise ValueError("Retrieval training requires two or more classes with positive pairs")
    batches = []
    for _ in range(math.ceil(len(dataset) / 32)):
        selected = rng.sample(groups, min(8, len(groups)))
        batches.append([i for indices in selected for i in rng.sample(indices, min(4, len(indices)))])
    return batches


def features(batch, device):
    return {key: batch[key].to(device) for key in ("visual", "text", "cue")}


def retrieval_loss(embedding, labels):
    logits = embedding @ embedding.T / 0.07
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    log_prob = logits.masked_fill(diagonal, -torch.inf).log_softmax(dim=1)
    return -(log_prob.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("hcvr.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    split = args.data / "split.json"
    tokens = args.data / "tokens"
    if not split.is_file() or not tokens.is_dir():
        parser.error("Provide split.json and tokens/<video_id>.pt under --data; each tensor file needs patches and sentence")
    spec = json.loads(split.read_text(encoding="utf-8"))
    if "pairs" in spec:
        dataset = ConViS(split, tokens, "train")
    elif {"records", "indices", "classes"} <= spec.keys():
        dataset = HCDMeta(split, tokens, "train")
    else:
        parser.error("Expected a category or video-pair split")
    if not len(dataset):
        parser.error("The training split is empty")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HCVR().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    for epoch in range(args.epochs):
        if isinstance(dataset, HCDMeta):
            loader = DataLoader(dataset, batch_sampler=class_batches(dataset, rng))
        else:
            loader = make_loader(dataset, 16, shuffle=True)
        total = 0.0
        for batch in loader:
            if isinstance(dataset, HCDMeta):
                loss = retrieval_loss(model(features(batch, device)), batch["label"].to(device))
            else:
                left = model(features(batch["video1"], device))
                right = model(features(batch["video2"], device))
                ratings = (batch["scores"].to(device) - 1) / 4
                preferences = ratings[:, None, :] - ratings[None, :, :]
                mask = preferences.abs() >= 0.125
                scores = (left * right).sum(1)
                gaps = scores[:, None, None] - scores[None, :, None]
                penalties = F.softplus(-preferences.sign() * gaps / 0.1)
                loss = penalties[mask].mean() if mask.any() else scores.sum() * 0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch + 1} loss={total / len(loader):.6f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
