
import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from utils import save_best_checkpoint


@dataclass
class EvaluationResult:
    auroc: float
    auprc: float
    accuracy: float
    loss: float
    threshold: float


def class_weights_from_loader(train_loader, device: str) -> torch.Tensor:
    labels = train_loader.dataset.df["Y"].astype(int).to_numpy()
    counts = np.maximum(np.bincount(labels, minlength=2).astype(np.float32), 1.0)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def linear_warmup(epoch: int, target: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return float(target)
    return float(target) * min(1.0, max(0.0, epoch / float(warmup_epochs)))


def geometric_temperature(epoch: int, args) -> float:
    start = float(args.vgib_temperature_start)
    end = float(args.vgib_temperature_end)
    anneal_epochs = int(args.vgib_temperature_anneal_epochs)
    if anneal_epochs <= 0 or start == end:
        return end
    ratio = min(1.0, max(0.0, epoch / float(anneal_epochs)))
    return start * math.pow(end / start, ratio)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    focal_gamma: float,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if focal_gamma > 0:
        with torch.no_grad():
            probability = F.softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
            focal_weight = (1.0 - probability.clamp(1e-6, 1.0)).pow(float(focal_gamma))
        losses = losses * focal_weight
    return losses.mean()


def vgib_loss(bridge_stats: Dict[str, torch.Tensor], args) -> torch.Tensor:
    probability = bridge_stats["gate_probability"]
    node_mask = bridge_stats["graph_node_mask"].bool()
    eps = float(args.vgib_kl_eps)
    posterior = probability.clamp(eps, 1.0 - eps)
    prior = posterior.new_full(posterior.shape, float(args.vgib_prior_keep)).clamp(
        eps, 1.0 - eps
    )
    kl_per_node = posterior * torch.log(posterior / prior) + (1.0 - posterior) * torch.log(
        (1.0 - posterior) / (1.0 - prior)
    )
    mask_float = node_mask.to(kl_per_node.dtype)
    rate = (kl_per_node * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
    return torch.clamp(rate, min=float(args.vgib_free_bits)).mean()


def bridge_contrastive_loss(
    bridge_stats: Dict[str, torch.Tensor], labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    positive_mask = labels.bool()
    if positive_mask.sum() == 0:
        return labels.new_tensor(0.0, dtype=torch.float32)
    targets = torch.arange(labels.size(0), device=labels.device)[positive_mask]
    logits_dp = (
        bridge_stats["drug_to_protein"][positive_mask]
        @ bridge_stats["protein_anchor"].t()
        / temperature
    )
    logits_pd = (
        bridge_stats["protein_to_drug"][positive_mask]
        @ bridge_stats["drug_anchor"].t()
        / temperature
    )
    return 0.5 * (F.cross_entropy(logits_dp, targets) + F.cross_entropy(logits_pd, targets))


def total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bridge_stats: Dict[str, torch.Tensor],
    epoch: int,
    args,
    class_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cls = classification_loss(logits, labels, class_weights, args.focal_gamma)
    vgib = vgib_loss(bridge_stats, args)
    beta = linear_warmup(epoch, args.vgib_beta, args.vgib_beta_warmup_epochs)
    bcr = bridge_contrastive_loss(
        bridge_stats, labels, args.bridge_contrast_temperature
    )
    gamma = linear_warmup(epoch, args.bridge_contrast_gamma, args.contrast_warmup_epochs)

    connectivity = bridge_stats["graph_connectivity_loss"]
    cross = F.relu(
        bridge_stats["graph_cross_edge_fraction"].new_tensor(args.graph_cross_target)
        - bridge_stats["graph_cross_edge_fraction"]
    ).mean()
    balance = bridge_stats["graph_balance_loss"]

    total = (
        cls
        + beta * vgib
        + gamma * bcr
        + args.graph_connect_lambda * connectivity
        + args.graph_cross_lambda * cross
        + args.graph_balance_lambda * balance
    )
    return total, {
        "total": float(total.detach()),
        "classification": float(cls.detach()),
        "vgib": float(vgib.detach()),
        "bcr": float(bcr.detach()),
        "beta": beta,
        "gamma": gamma,
        "posterior": float(bridge_stats["retain_mean"].mean().detach()),
        "temperature": float(bridge_stats["vgib_temperature"].detach()),
    }


def move_batch(batch, device: str):
    drug_batch, protein_batch, labels = batch
    drug_batch = tuple(tensor.to(device) for tensor in drug_batch)
    protein_batch = tuple(tensor.to(device) for tensor in protein_batch)
    return drug_batch, protein_batch, labels.long().to(device)


def train_one_epoch(
    model,
    optimizer,
    dataloader,
    epoch: int,
    args,
    class_weights: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.train()
    model.bridge.set_temperature(geometric_temperature(epoch, args))
    logs = []
    for batch in tqdm(dataloader, desc="Train {}".format(epoch + 1)):
        optimizer.zero_grad(set_to_none=True)
        drug_batch, protein_batch, labels = move_batch(batch, args.device)
        logits, bridge_stats = model(drug_batch, protein_batch)
        loss, diagnostics = total_loss(
            logits, labels, bridge_stats, epoch, args, class_weights
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        logs.append(diagnostics)
    return {key: float(np.mean([entry[key] for entry in logs])) for key in logs[0]}


def collect_predictions(model, dataloader, device: str):
    model.eval()
    labels, probabilities, losses = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            drug_batch, protein_batch, batch_labels = move_batch(batch, device)
            logits, _ = model(drug_batch, protein_batch)
            probability = F.softmax(logits, dim=-1)[:, 1]
            losses.append(float(F.cross_entropy(logits, batch_labels)))
            labels.extend(batch_labels.cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.unique(labels).size < 2:
        raise ValueError("AUROC/AUPRC require both classes.")
    return labels, probabilities, float(np.mean(losses))


def ranking_metrics(model, dataloader, device: str) -> Tuple[float, float, float]:
    labels, probabilities, loss = collect_predictions(model, dataloader, device)
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
        loss,
    )


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    validation_loader,
    logger,
    args,
    class_weights: Optional[torch.Tensor],
) -> float:
    best_auroc = -1.0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        logs = train_one_epoch(
            model, optimizer, train_loader, epoch, args, class_weights
        )
        val_auroc, val_auprc, val_loss = ranking_metrics(
            model, validation_loader, args.device
        )
        scheduler.step(val_auroc)
        logger.info(
            "Epoch %d/%d | loss=%.4f cls=%.4f VGIB=%.4f BCR=%.4f "
            "beta=%.6f gamma=%.6f post=%.4f temp=%.4f | "
            "val AUROC=%.4f AUPRC=%.4f loss=%.4f",
            epoch + 1,
            args.epochs,
            logs["total"],
            logs["classification"],
            logs["vgib"],
            logs["bcr"],
            logs["beta"],
            logs["gamma"],
            logs["posterior"],
            logs["temperature"],
            val_auroc,
            val_auprc,
            val_loss,
        )
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            epochs_without_improvement = 0
            save_best_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "best_auroc": best_auroc,
                    "args": vars(args),
                },
                args.model_save_dir,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break
    return best_auroc


def select_accuracy_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> float:
    thresholds = np.unique(probabilities)

    if thresholds.size == 0:
        return 0.5

    accuracies = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int64)
        accuracies.append(
            accuracy_score(labels, predictions)
        )

    return float(thresholds[int(np.argmax(accuracies))])


def evaluate_with_validation_threshold(
    model, validation_loader, test_loader, args
) -> EvaluationResult:

    val_labels, val_probabilities, _ = collect_predictions(model, validation_loader, args.device)

    threshold = select_accuracy_threshold(val_labels,val_probabilities,)

    test_labels, test_probabilities, test_loss = collect_predictions(
        model,
        test_loader,
        args.device,
    )

    predictions = (test_probabilities >= threshold).astype(np.int64)

    return EvaluationResult(
        auroc=float(roc_auc_score(test_labels, test_probabilities)),
        auprc=float(average_precision_score(test_labels, test_probabilities)),
        accuracy=float(accuracy_score(test_labels, predictions)),
        loss=float(test_loss),
        threshold=float(threshold),
    )


def result_to_dict(result: EvaluationResult) -> Dict[str, float]:
    return asdict(result)
