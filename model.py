
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge import StrictVariationalGraphBridge
from layers import (
    BidirectionalCrossAttention,
    BridgeTransform,
    InteractionEvidencePooler,
    Projector,
    StableLayerNorm,
)


class CFGBDTI(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.interaction_residual_scale = float(args.interaction_residual_scale)

        # Foundation Projection
        self.drug_projector = Projector(args.drug_pretrained_dim, args.d_model, args.dropout)
        self.protein_projector = Projector(
            args.protein_pretrained_dim, args.d_model, args.dropout
        )

        # Bidirectional Token-Level Interaction (BTI)
        self.interaction = BidirectionalCrossAttention(
            args.d_model, args.n_heads, args.dropout, talking_heads=args.talking_heads
        )

        # Interaction-Guided Pooling (IGP)
        self.drug_pooler = InteractionEvidencePooler(
            args.d_model, args.evidence_scale, args.dropout
        )
        self.protein_pooler = InteractionEvidencePooler(
            args.d_model, args.evidence_scale, args.dropout
        )

        # Variational Graph Bottleneck Bridge (VGBB)
        self.bridge = StrictVariationalGraphBridge(
            d_model=args.d_model,
            ib_dim=args.ib_dim,
            dropout=args.dropout,
            topk_drug=args.graph_bridge_topk_drug,
            topk_protein=args.graph_bridge_topk_protein,
            num_layers=args.graph_bridge_layers,
            adjacency_temperature=args.graph_bridge_adj_temperature,
            selector_context_layers=args.vgib_selector_context_layers,
            initial_keep_probability=args.vgib_init_keep,
            temperature_start=args.vgib_temperature_start,
        )

        # Bridge Contrastive Regularization (BCR)
        self.drug_to_protein_bridge = BridgeTransform(
            args.d_model, args.ib_dim, args.dropout
        )
        self.protein_to_drug_bridge = BridgeTransform(
            args.d_model, args.ib_dim, args.dropout
        )
        self.drug_anchor_projection = nn.Sequential(
            StableLayerNorm(args.d_model), nn.Linear(args.d_model, args.d_model)
        )
        self.protein_anchor_projection = nn.Sequential(
            StableLayerNorm(args.d_model), nn.Linear(args.d_model, args.d_model)
        )

        # Prediction Head
        classifier_hidden = max(2 * args.d_model, 2 * args.ib_dim)
        self.classifier = nn.Sequential(
            StableLayerNorm(args.ib_dim),
            nn.Linear(args.ib_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(classifier_hidden, args.n_class),
        )

    # Interaction-Guided Bridge Graph Construction
    @staticmethod
    def _token_evidence(
        drug_attention: torch.Tensor,
        protein_attention: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        drug_evidence = drug_attention.mean(dim=1).amax(dim=-1)
        protein_evidence = protein_attention.mean(dim=1).amax(dim=-2)
        return (
            drug_evidence.masked_fill(~drug_mask.bool(), 0.0),
            protein_evidence.masked_fill(~protein_mask.bool(), 0.0),
        )

    def _encode_pair(self, drug_batch, protein_batch) -> Dict[str, torch.Tensor]:
        drug_embedding, drug_mask = drug_batch
        protein_embedding, protein_mask = protein_batch
        drug_mask, protein_mask = drug_mask.bool(), protein_mask.bool()

        # Foundation Projection
        drug_projected = self.drug_projector(drug_embedding)
        protein_projected = self.protein_projector(protein_embedding)
        # Bidirectional Token-Level Interaction (BTI)
        drug_context, protein_context, drug_attention, protein_attention = self.interaction(
            drug_projected, protein_projected, drug_mask, protein_mask
        )
        drug_tokens = drug_projected + self.interaction_residual_scale * drug_context
        protein_tokens = protein_projected + self.interaction_residual_scale * protein_context
        # Cross-Foundation Token Relevance
        drug_evidence, protein_evidence = self._token_evidence(
            drug_attention, protein_attention, drug_mask, protein_mask
        )
        # Interaction-Guided Pooling (IGP)
        drug_pooled, _ = self.drug_pooler(drug_tokens, drug_mask, drug_evidence)
        protein_pooled, _ = self.protein_pooler(
            protein_tokens, protein_mask, protein_evidence
        )
        return {
            "drug_tokens": drug_tokens,
            "protein_tokens": protein_tokens,
            "drug_mask": drug_mask,
            "protein_mask": protein_mask,
            "drug_evidence": drug_evidence,
            "protein_evidence": protein_evidence,
            "drug_attention": drug_attention,
            "protein_attention": protein_attention,
            "drug_pooled": drug_pooled,
            "protein_pooled": protein_pooled,
        }

    def forward(self, drug_batch, protein_batch):
        encoded = self._encode_pair(drug_batch, protein_batch)

        # Variational Graph Bottleneck Bridge (VGBB)
        bridge_representation, bridge_stats = self.bridge(
            drug_tokens=encoded["drug_tokens"],
            protein_tokens=encoded["protein_tokens"],
            drug_mask=encoded["drug_mask"],
            protein_mask=encoded["protein_mask"],
            drug_evidence=encoded["drug_evidence"],
            protein_evidence=encoded["protein_evidence"],
            drug_attention=encoded["drug_attention"],
            protein_attention=encoded["protein_attention"],
        )
        # Prediction Head
        logits = self.classifier(bridge_representation)

        # Bridge Contrastive Regularization (BCR)
        bridge_stats.update(
            {
                "drug_anchor": F.normalize(
                    self.drug_anchor_projection(encoded["drug_pooled"]), dim=-1, eps=1e-6
                ),
                "protein_anchor": F.normalize(
                    self.protein_anchor_projection(encoded["protein_pooled"]),
                    dim=-1,
                    eps=1e-6,
                ),
                "drug_to_protein": F.normalize(
                    self.drug_to_protein_bridge(
                        encoded["drug_pooled"], bridge_representation
                    ),
                    dim=-1,
                    eps=1e-6,
                ),
                "protein_to_drug": F.normalize(
                    self.protein_to_drug_bridge(
                        encoded["protein_pooled"], bridge_representation
                    ),
                    dim=-1,
                    eps=1e-6,
                ),
            }
        )
        return logits, bridge_stats
