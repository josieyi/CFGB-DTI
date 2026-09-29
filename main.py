
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import create_dataloaders
from model import CFGBDTI
from training import class_weights_from_loader, evaluate_with_validation_threshold, train
from utils import create_logger, set_seed


DRUG_DIMENSIONS = {"smiles": 768, "unimol": 512}
PROTEIN_DIMENSIONS = {"prott5": 1024, "saprot": 1280}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", type=str_to_bool, default=True)

    parser.add_argument("--dataset", choices=["bindingdb", "biosnap", "human"], required=True)
    parser.add_argument("--split", choices=["random", "cold", "cluster"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--allow_cluster_test_as_val", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=5)

    parser.add_argument("--drug_view", choices=["smiles", "unimol"], default="smiles")
    parser.add_argument("--protein_view", choices=["prott5", "saprot"], default="prott5")
    parser.add_argument("--drug_pretrained_dim", type=int, default=0)
    parser.add_argument("--protein_pretrained_dim", type=int, default=0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--ib_dim", type=int, default=256)
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--talking_heads", type=str_to_bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--interaction_residual_scale", type=float, default=0.5)
    parser.add_argument("--evidence_scale", type=float, default=0.5)

    parser.add_argument("--graph_bridge_topk_drug", type=int, default=32)
    parser.add_argument("--graph_bridge_topk_protein", type=int, default=32)
    parser.add_argument("--graph_bridge_layers", type=int, default=1)
    parser.add_argument("--graph_bridge_adj_temperature", type=float, default=0.2)
    parser.add_argument("--vgib_selector_context_layers", type=int, default=1)
    parser.add_argument("--vgib_beta", type=float, default=0.002)
    parser.add_argument("--vgib_beta_warmup_epochs", type=int, default=20)
    parser.add_argument("--vgib_prior_keep", type=float, default=0.2)
    parser.add_argument("--vgib_init_keep", type=float, default=0.8)
    parser.add_argument("--vgib_temperature_start", type=float, default=1.0)
    parser.add_argument("--vgib_temperature_end", type=float, default=0.3)
    parser.add_argument("--vgib_temperature_anneal_epochs", type=int, default=40)
    parser.add_argument("--vgib_free_bits", type=float, default=0.0)
    parser.add_argument("--vgib_kl_eps", type=float, default=1e-6)

    parser.add_argument("--graph_connect_lambda", type=float, default=0.0)
    parser.add_argument("--graph_cross_lambda", type=float, default=0.0)
    parser.add_argument("--graph_cross_target", type=float, default=0.08)
    parser.add_argument("--graph_balance_lambda", type=float, default=0.0)
    parser.add_argument("--bridge_contrast_gamma", type=float, default=0.0)
    parser.add_argument("--bridge_contrast_temperature", type=float, default=0.1)
    parser.add_argument("--contrast_warmup_epochs", type=int, default=30)

    parser.add_argument("--use_class_weight", type=str_to_bool, default=True)
    parser.add_argument("--focal_gamma", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--core_lr_scale", type=float, default=0.5)
    parser.add_argument("--optimizer_eps", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--result_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.d_model % args.n_heads != 0:
        parser.error("d_model must be divisible by n_heads.")
    if args.drug_pretrained_dim <= 0:
        args.drug_pretrained_dim = DRUG_DIMENSIONS[args.drug_view]
    if args.protein_pretrained_dim <= 0:
        args.protein_pretrained_dim = PROTEIN_DIMENSIONS[args.protein_view]
    return args


def build_optimizer(model, args):
    core_keywords = (
        "drug_projector",
        "protein_projector",
        "interaction",
        "drug_pooler",
        "protein_pooler",
        "bridge",
    )
    core_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(keyword in name for keyword in core_keywords):
            core_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    return AdamW(
        [
            {"params": core_parameters, "lr": args.lr * args.core_lr_scale},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        eps=args.optimizer_eps,
    )


def main() -> None:
    args = parse_arguments()
    set_seed(args.seed, args.deterministic)
    run_name = "cfgb_{}_{}_{}_{}_seed{}".format(
        args.drug_view, args.protein_view, args.dataset, args.split, args.seed
    )
    args.log_dir = args.log_dir / run_name
    args.model_save_dir = args.checkpoint_dir / run_name
    args.result_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(args.log_dir)
    logger.info("Arguments: %s", vars(args))

    train_loader, validation_loader, test_loader = create_dataloaders(args)
    model = CFGBDTI(args).to(args.device)
    optimizer = build_optimizer(model, args)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-8
    )
    class_weights = (
        class_weights_from_loader(train_loader, args.device)
        if args.use_class_weight
        else None
    )

    train(
        model,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
        logger,
        args,
        class_weights,
    )

    best_path = args.model_save_dir / "model_best.pth"
    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    result = evaluate_with_validation_threshold(
        model, validation_loader, test_loader, args
    )

    result_text = (
        "AUROC={:.6f}, AUPRC={:.6f}, F1={:.6f}, BestValAUROC={:.6f}".format(
            result.auroc, result.auprc, result.f1, float(checkpoint["best_auroc"])
        )
    )
    result_path = args.result_dir / "{}_{}.txt".format(args.dataset, args.split)
    with result_path.open("a", encoding="utf-8") as file:
        file.write("RunName: {}\n{}\n{}\n".format(run_name, result_text, "=" * 80))
    logger.info("Test result: %s", result_text)


if __name__ == "__main__":
    main()
