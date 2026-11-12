import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def load_features(directory, video_id):
    if not str(video_id) or Path(str(video_id)).name != str(video_id) or "\\" in str(video_id) or ":" in str(video_id):
        raise ValueError("Expected a video identifier without directory components")
    data = torch.load(directory / f"{video_id}.pt", map_location="cpu", weights_only=True)
    visual = F.normalize(data["patches"][:, 0].mean(0), dim=-1)
    text = F.normalize(data["sentence"], dim=-1)
    return {"visual": visual, "text": text, "cue": text}


class HCDMeta(Dataset):
    def __init__(self, split_file, token_dir, split):
        spec = json.loads(Path(split_file).read_text(encoding="utf-8"))
        self.directory = Path(token_dir)
        self.records = [spec["records"][i] for i in spec["indices"][split]]
        self.labels = {name: i for i, name in enumerate(spec["classes"])}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {**load_features(self.directory, row["video_id"]), "video_id": row["video_id"],
                "label": self.labels[row["keyword"]]}


class ConViS(Dataset):
    concepts = ("MainAction", "MainSubjects", "MainObjects", "Location", "OrderOfActions")

    def __init__(self, split_file, token_dir, split):
        self.directory = Path(token_dir)
        self.pairs = json.loads(Path(split_file).read_text(encoding="utf-8"))["pairs"][split]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row = self.pairs[index]
        a, b = (Path(row[key]).stem for key in ("video1", "video2"))
        return {"video1": load_features(self.directory, a), "video2": load_features(self.directory, b),
                "id1": a, "id2": b, "scores": torch.tensor([row[key] for key in self.concepts],
                                                           dtype=torch.float32)}


def make_loader(dataset, batch_size, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
