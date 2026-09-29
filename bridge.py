
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from layers import DenseGraphLayer, StableLayerNorm


class StrictVariationalGraphBridge(nn.Module):
    def __init__(
        self,
        d_model: int,
        ib_dim: int,
        dropout: float,
        topk_drug: int,
        topk_protein: int,
        num_layers: int,
        adjacency_temperature: float,
        selector_context_layers: int,
        initial_keep_probability: float,
        temperature_start: float,
    ) -> None:
        super().__init__()
        if topk_drug <= 0 or topk_protein <= 0 or num_layers <= 0:
            raise ValueError("Graph sizes and layer count must be positive.")
        if adjacency_temperature <= 0 or temperature_start <= 0:
            raise ValueError("Temperatures must be positive.")

        self.topk_drug = int(topk_drug)
        self.topk_protein = int(topk_protein)
        self.adjacency_temperature = float(adjacency_temperature)

        self.side_embedding = nn.Parameter(torch.empty(2, d_model))
        nn.init.normal_(self.side_embedding, mean=0.0, std=0.02)
        self.node_norm = StableLayerNorm(d_model)
        self.selector_layers = nn.ModuleList(
            [DenseGraphLayer(d_model, dropout) for _ in range(selector_context_layers)]
        )
        self.prediction_layers = nn.ModuleList(
            [DenseGraphLayer(d_model, dropout) for _ in range(num_layers)]
        )

        selector_hidden = max(1, d_model // 2)
        self.gate_network = nn.Sequential(
            StableLayerNorm(d_model),
            nn.Linear(d_model, selector_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(selector_hidden, 1),
        )
        nn.init.normal_(self.gate_network[-1].weight, mean=0.0, std=1e-3)
        initial_logit = math.log(initial_keep_probability / (1.0 - initial_keep_probability))
        nn.init.constant_(self.gate_network[-1].bias, initial_logit)

        bridge_hidden = max(2 * d_model, 2 * ib_dim)
        self.bridge_encoder = nn.Sequential(
            StableLayerNorm(d_model),
            nn.Linear(d_model, bridge_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bridge_hidden, ib_dim),
            StableLayerNorm(ib_dim),
        )
        self.register_buffer(
            "current_temperature", torch.tensor(float(temperature_start), dtype=torch.float32)
        )

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("Binary-Concrete temperature must be positive.")
        self.current_temperature.fill_(float(temperature))

    @staticmethod
    def _masked_topk(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        scores: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, length, feature_dim = tokens.shape
        k = max(1, min(int(k), int(length)))
        mask = mask.bool()
        masked_scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
        indices = masked_scores.topk(k, dim=-1).indices
        selected_tokens = torch.gather(
            tokens, 1, indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        )
        selected_mask = torch.gather(mask, 1, indices)
        selected_tokens = selected_tokens * selected_mask.unsqueeze(-1).to(selected_tokens.dtype)
        return selected_tokens, selected_mask, indices

    @staticmethod
    def _valid_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
        mask_float = mask.to(values.dtype)
        return (values * mask_float).sum(dim=dim) / mask_float.sum(dim=dim).clamp_min(1.0)

    @staticmethod
    def _gather_cross_edges(
        drug_to_protein: torch.Tensor,
        protein_to_drug_transposed: torch.Tensor,
        drug_indices: torch.Tensor,
        protein_indices: torch.Tensor,
    ) -> torch.Tensor:
        drug_map = drug_to_protein.mean(dim=1)
        protein_map = protein_to_drug_transposed.mean(dim=1)
        kd = drug_indices.size(1)
        row_index = drug_indices.unsqueeze(-1).expand(-1, -1, drug_map.size(-1))
        drug_map = torch.gather(drug_map, 1, row_index)
        protein_map = torch.gather(protein_map, 1, row_index)
        column_index = protein_indices.unsqueeze(1).expand(-1, kd, -1)
        drug_map = torch.gather(drug_map, 2, column_index)
        protein_map = torch.gather(protein_map, 2, column_index)
        return 0.5 * (drug_map + protein_map)

    def _build_candidate_graph(
        self,
        drug_nodes: torch.Tensor,
        protein_nodes: torch.Tensor,
        drug_valid: torch.Tensor,
        protein_valid: torch.Tensor,
        drug_indices: torch.Tensor,
        protein_indices: torch.Tensor,
        cross_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, kd, _ = drug_nodes.shape
        kp = protein_nodes.size(1)
        dtype, device = drug_nodes.dtype, drug_nodes.device

        drug_nodes = drug_nodes + self.side_embedding[0].view(1, 1, -1).to(dtype)
        protein_nodes = protein_nodes + self.side_embedding[1].view(1, 1, -1).to(dtype)
        nodes = self.node_norm(torch.cat([drug_nodes, protein_nodes], dim=1))
        node_mask = torch.cat([drug_valid, protein_valid], dim=1).bool()
        adjacency = torch.zeros(batch_size, kd + kp, kd + kp, dtype=dtype, device=device)

        drug_distance = (drug_indices.float().unsqueeze(2) - drug_indices.float().unsqueeze(1)).abs()
        drug_adj = torch.exp(
            -drug_distance / max(self.adjacency_temperature * max(kd, 1), 1e-6)
        ).to(dtype)
        drug_adj *= (drug_valid.unsqueeze(2) & drug_valid.unsqueeze(1)).to(dtype)
        adjacency[:, :kd, :kd] = drug_adj

        protein_distance = (
            protein_indices.float().unsqueeze(2) - protein_indices.float().unsqueeze(1)
        ).abs()
        protein_adj = torch.exp(
            -protein_distance / max(self.adjacency_temperature * max(kp, 1), 1e-6)
        ).to(dtype)
        protein_adj *= (protein_valid.unsqueeze(2) & protein_valid.unsqueeze(1)).to(dtype)
        adjacency[:, kd:, kd:] = protein_adj

        cross_mask = drug_valid.unsqueeze(2) & protein_valid.unsqueeze(1)
        cross_edges = cross_edges.to(dtype) * cross_mask.to(dtype)
        adjacency[:, :kd, kd:] = cross_edges
        adjacency[:, kd:, :kd] = cross_edges.transpose(1, 2)

        identity = torch.eye(kd + kp, dtype=dtype, device=device).unsqueeze(0)
        adjacency = adjacency + identity * node_mask.unsqueeze(1).to(dtype)
        valid_pairs = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        adjacency = adjacency * valid_pairs.to(dtype)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        nodes = nodes * node_mask.unsqueeze(-1).to(dtype)
        return nodes, adjacency, node_mask

    def _selector_context(
        self, nodes: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        context = nodes
        for layer in self.selector_layers:
            context = layer(context, adjacency, node_mask)
        return context

    @staticmethod
    def _binary_concrete_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((logits + logistic_noise) / temperature)

    def _posterior_and_gate(
        self, selector_context: torch.Tensor, node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate_network(selector_context).squeeze(-1)
        logits = logits.masked_fill(~node_mask, 0.0)
        posterior = torch.sigmoid(logits) * node_mask.to(logits.dtype)
        if self.training:
            gate = self._binary_concrete_sample(
                logits, float(self.current_temperature.item())
            ) * node_mask.to(logits.dtype)
        else:
            gate = posterior
        return posterior, gate

    @staticmethod
    def _sender_gated_adjacency(
        adjacency: torch.Tensor, gate: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        valid_pairs = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        return adjacency * gate.unsqueeze(1) * valid_pairs.to(adjacency.dtype)

    @staticmethod
    def _connectivity_loss(
        posterior: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        assignment = torch.stack([posterior, 1.0 - posterior], dim=-1)
        assignment *= node_mask.unsqueeze(-1).to(assignment.dtype)
        cluster_adjacency = torch.einsum(
            "bni,bnm,bmj->bij", assignment, adjacency, assignment
        )
        cluster_adjacency /= cluster_adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        identity = torch.eye(2, device=posterior.device, dtype=posterior.dtype)
        return (cluster_adjacency - identity.unsqueeze(0)).pow(2).sum((1, 2)).mean()

    @staticmethod
    def _cross_edge_fraction(
        posterior: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor, kd: int
    ) -> torch.Tensor:
        pair_keep = posterior.unsqueeze(2) * posterior.unsqueeze(1)
        valid_pairs = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        total_mass = (pair_keep * adjacency * valid_pairs.to(adjacency.dtype)).sum((1, 2)).clamp_min(1e-12)
        cross_mask = torch.zeros_like(adjacency, dtype=torch.bool)
        cross_mask[:, :kd, kd:] = True
        cross_mask[:, kd:, :kd] = True
        cross_mass = (
            pair_keep * adjacency * (cross_mask & valid_pairs).to(adjacency.dtype)
        ).sum((1, 2))
        return cross_mass / total_mass

    @staticmethod
    def _side_balance(
        posterior: torch.Tensor, drug_valid: torch.Tensor, protein_valid: torch.Tensor
    ) -> torch.Tensor:
        kd = drug_valid.size(1)
        drug_mean = StrictVariationalGraphBridge._valid_mean(
            posterior[:, :kd], drug_valid, dim=1
        )
        protein_mean = StrictVariationalGraphBridge._valid_mean(
            posterior[:, kd:], protein_valid, dim=1
        )
        return (drug_mean - protein_mean).abs().mean()

    def _message_pass(
        self, nodes: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        representation = nodes
        for layer in self.prediction_layers:
            representation = layer(representation, adjacency, node_mask)
        return representation

    def forward(
        self,
        drug_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
        drug_mask: torch.Tensor,
        protein_mask: torch.Tensor,
        drug_evidence: torch.Tensor,
        protein_evidence: torch.Tensor,
        drug_attention: torch.Tensor,
        protein_attention: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        drug_nodes, drug_valid, drug_indices = self._masked_topk(
            drug_tokens, drug_mask, drug_evidence, self.topk_drug
        )
        protein_nodes, protein_valid, protein_indices = self._masked_topk(
            protein_tokens, protein_mask, protein_evidence, self.topk_protein
        )
        cross_edges = self._gather_cross_edges(
            drug_attention, protein_attention, drug_indices, protein_indices
        )
        nodes, adjacency, node_mask = self._build_candidate_graph(
            drug_nodes,
            protein_nodes,
            drug_valid,
            protein_valid,
            drug_indices,
            protein_indices,
            cross_edges,
        )

        selector_context = self._selector_context(nodes, adjacency, node_mask)
        posterior, gate = self._posterior_and_gate(selector_context, node_mask)
        gated_adjacency = self._sender_gated_adjacency(adjacency, gate, node_mask)
        selected_nodes = self._message_pass(nodes, gated_adjacency, node_mask)

        numerator = (
            gate.unsqueeze(-1)
            * selected_nodes
            * node_mask.unsqueeze(-1).to(selected_nodes.dtype)
        ).sum(dim=1)
        denominator = (gate * node_mask.to(gate.dtype)).sum(dim=1, keepdim=True).clamp_min(1e-6)
        selected_feature = numerator / denominator
        bridge_representation = self.bridge_encoder(selected_feature)

        return bridge_representation, {
            "gate_probability": posterior,
            "graph_node_mask": node_mask,
            "retain_mean": self._valid_mean(posterior, node_mask, dim=1),
            "sample_retain_mean": self._valid_mean(gate, node_mask, dim=1),
            "graph_connectivity_loss": self._connectivity_loss(
                posterior, adjacency, node_mask
            ),
            "graph_cross_edge_fraction": self._cross_edge_fraction(
                posterior, adjacency, node_mask, drug_nodes.size(1)
            ),
            "graph_balance_loss": self._side_balance(
                posterior, drug_valid, protein_valid
            ),
            "vgib_temperature": posterior.new_tensor(float(self.current_temperature.item())),
        }
