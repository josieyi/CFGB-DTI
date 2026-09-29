
import torch
import torch.nn as nn
from torch import einsum
from typing import Tuple, Union


class StableLayerNorm(nn.Module):

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
        y = x.to(torch.float64)
        mean = y.mean(dim=dims, keepdim=True)
        var = (y - mean).pow(2).mean(dim=dims, keepdim=True)
        y = ((y - mean) / torch.sqrt(var + self.eps)).to(dtype=x.dtype)
        if self.elementwise_affine:
            y = y * self.weight.to(dtype=x.dtype) + self.bias.to(dtype=x.dtype)
        return y


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Numerically safe softmax over valid token pairs only."""
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
    probabilities = torch.softmax(masked_logits, dim=dim)
    probabilities = probabilities * mask.to(logits.dtype)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-12)


# Foundation Projection
class Projector(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            StableLayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            StableLayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Bidirectional Token-Level Interaction (BTI)
class BidirectionalCrossAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        talking_heads: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        inner_dim = self.n_heads * self.head_dim

        self.drug_norm = StableLayerNorm(d_model)
        self.protein_norm = StableLayerNorm(d_model)
        self.drug_to_qk = nn.Linear(d_model, inner_dim, bias=False)
        self.protein_to_qk = nn.Linear(d_model, inner_dim, bias=False)
        self.drug_to_value = nn.Linear(d_model, inner_dim, bias=False)
        self.protein_to_value = nn.Linear(d_model, inner_dim, bias=False)
        self.drug_output = nn.Linear(inner_dim, d_model)
        self.protein_output = nn.Linear(inner_dim, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.head_mixer = (
            nn.Conv2d(n_heads, n_heads, kernel_size=1, bias=False)
            if talking_heads
            else nn.Identity()
        )

        # The interaction branch starts as an identity residual path.
        nn.init.zeros_(self.drug_output.weight)
        nn.init.zeros_(self.drug_output.bias)
        nn.init.zeros_(self.protein_output.weight)
        nn.init.zeros_(self.protein_output.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.shape
        return x.reshape(batch_size, length, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)

    def forward(
        self,
        drug_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_mask = drug_mask.bool()
        protein_mask = protein_mask.bool()
        drug_normalized = self.drug_norm(drug_tokens)
        protein_normalized = self.protein_norm(protein_tokens)

        drug_qk = self._split_heads(self.drug_to_qk(drug_normalized))
        protein_qk = self._split_heads(self.protein_to_qk(protein_normalized))
        drug_value = self._split_heads(self.drug_to_value(drug_normalized))
        protein_value = self._split_heads(self.protein_to_value(protein_normalized))

        # Eq. (attention_score): R in R^{B x H x L_d x L_p}.
        affinity = einsum(
            "b h i d, b h j d -> b h i j", drug_qk, protein_qk
        ) * self.scale
        affinity = self.head_mixer(affinity)

        pair_mask = drug_mask[:, None, :, None] & protein_mask[:, None, None, :]
        drug_to_protein = masked_softmax(affinity, pair_mask, dim=-1)
        # Stored with shape [B, H, L_d, L_p]; entry [i,j] equals M^{p->d}_{j,i}.
        protein_to_drug_transposed = masked_softmax(affinity, pair_mask, dim=-2)

        drug_context = einsum(
            "b h i j, b h j d -> b h i d",
            self.attention_dropout(drug_to_protein),
            protein_value,
        )
        protein_context = einsum(
            "b h j i, b h j d -> b h i d",
            self.attention_dropout(protein_to_drug_transposed),
            drug_value,
        )

        drug_context = self.drug_output(self._merge_heads(drug_context))
        protein_context = self.protein_output(self._merge_heads(protein_context))
        return (
            drug_context,
            protein_context,
            drug_to_protein,
            protein_to_drug_transposed,
        )


# Interaction-Guided Pooling (IGP)
class InteractionEvidencePooler(nn.Module):

    def __init__(self, d_model: int, evidence_scale: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(1, d_model // 2)
        self.scorer = nn.Sequential(
            StableLayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output = nn.Sequential(
            StableLayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            StableLayerNorm(d_model),
        )
        self.evidence_scale = float(evidence_scale)

    @staticmethod
    def _validate_mask(mask: torch.Tensor) -> None:
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape [B, L], got {tuple(mask.shape)}")
        if (mask.bool().sum(dim=1) == 0).any():
            raise ValueError("All-masked token sequences are not supported.")

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _normalize_evidence(
        evidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        evidence = evidence.masked_fill(~mask, 0.0)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = evidence.sum(dim=-1, keepdim=True) / count
        centered = (evidence - mean).masked_fill(~mask, 0.0)
        variance = centered.pow(2).sum(dim=-1, keepdim=True) / count
        return centered / torch.sqrt(variance + 1e-6)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        evidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_mask(mask)
        mask = mask.bool()
        mean_feature = self._masked_mean(tokens, mask)
        scores = self.scorer(tokens).squeeze(-1)
        scores = scores + self.evidence_scale * self._normalize_evidence(evidence, mask)
        weights = masked_softmax(scores, mask, dim=-1)
        evidence_feature = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
        pooled = self.output(torch.cat([mean_feature, evidence_feature], dim=-1))
        return pooled, weights


# Graph Propagation in VGBB
class DenseGraphLayer(nn.Module):

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.pre_norm = StableLayerNorm(d_model)
        self.update = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.post_norm = StableLayerNorm(d_model)

    def forward(
        self,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.pre_norm(nodes)
        messages = torch.bmm(adjacency, normalized)
        delta = self.update(torch.cat([normalized, messages], dim=-1))
        output = self.post_norm(nodes + delta)
        return output * node_mask.unsqueeze(-1).to(output.dtype)


# Bridge Contrastive Regularization (BCR)
class BridgeTransform(nn.Module):

    def __init__(self, d_model: int, ib_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            StableLayerNorm(d_model + ib_dim),
            nn.Linear(d_model + ib_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = StableLayerNorm(d_model)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, source: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        delta = self.network(torch.cat([source, bridge], dim=-1))
        return self.output_norm(source + delta)
