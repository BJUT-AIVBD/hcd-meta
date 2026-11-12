import torch
from torch import nn
from torch.nn import functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalProjector(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.visual = MLPBlock(visual_dim, model_dim, model_dim, dropout)
        self.text = MLPBlock(text_dim, model_dim, model_dim, dropout)

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"visual": self.visual(visual), "text": self.text(text)}


class HCVR(nn.Module):
    def __init__(self, dim=64, hidden=128, heads=4, layers=1, dropout=0.1):
        super().__init__()
        self.projector = MultimodalProjector(512, 512, dim, dropout)
        self.shared = MLPBlock(dim, hidden, dim, dropout)
        self.specific_visual = MLPBlock(dim, hidden, dim, dropout)
        self.specific_text = MLPBlock(dim, hidden, dim, dropout)
        self.cue_encoder = MLPBlock(512, dim, dim, dropout)
        self.global_projection = nn.Linear(4 * dim, dim)
        self.pool_queries = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(dim, heads, 256, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(), nn.LayerNorm(512))

    def forward(self, batch):
        visual = batch["visual"]
        text = batch["text"]
        projected = self.projector(visual, text)
        v, t = projected["visual"], projected["text"]
        content = torch.stack((self.shared(v), self.shared(t), self.specific_visual(v),
                               self.specific_text(t)), dim=1)
        global_token = self.global_projection(content.flatten(1)).unsqueeze(1)
        queries = self.pool_queries.unsqueeze(0).expand(content.size(0), -1, -1)
        sequence = torch.cat((global_token, queries, content), dim=1)
        cue = self.cue_encoder(batch.get("cue", text)).unsqueeze(1)
        attended, _ = self.cross_attention(sequence + cue, content, content)
        pooled = self.transformer(sequence + attended)[:, 0]
        branch = F.normalize(self.output(pooled), dim=-1)
        joint = F.normalize(F.normalize(visual, dim=-1) + F.normalize(text, dim=-1), dim=-1)
        return F.normalize(joint + 0.25 * branch, dim=-1)
