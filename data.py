
import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils import collate_pairs


DRUG_VIEW_TO_DIRECTORY = {
    "smiles": "smiles_embeddings",
    "unimol": "unimol_embeddings",
}
PROTEIN_VIEW_TO_DIRECTORY = {
    "prott5": "protein_embeddings",
    "saprot": "structure_embeddings",
}


class DTIDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset: str,
        drug_view: str,
        protein_view: str,
        embedding_root: str,
    ) -> None:
        required = {"SMILES_id", "Protein_id", "Y"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError("Missing CSV columns: {}".format(sorted(missing)))
        self.dataframe = dataframe.reset_index(drop=True)
        self.df = self.dataframe
        self.drug_directory = os.path.join(
            embedding_root, dataset, DRUG_VIEW_TO_DIRECTORY[drug_view]
        )
        self.protein_directory = os.path.join(
            embedding_root, dataset, PROTEIN_VIEW_TO_DIRECTORY[protein_view]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        drug_embedding, drug_mask = self._load_embedding(
            os.path.join(self.drug_directory, "{}.pt".format(row["SMILES_id"]))
        )
        protein_embedding, protein_mask = self._load_embedding(
            os.path.join(self.protein_directory, "{}.pt".format(row["Protein_id"]))
        )
        return drug_embedding, protein_embedding, drug_mask, protein_mask, int(row["Y"])

    @staticmethod
    def _load_embedding(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        record = torch.load(path, map_location="cpu")
        if "seq_emb" in record and "mask" in record:
            embedding, mask = record["seq_emb"], record["mask"]
        elif "embedding" in record and "mask" in record:
            embedding, mask = record["embedding"], record["mask"]
        else:
            raise KeyError("Expected seq_emb/mask or embedding/mask in {}".format(path))

        embedding = torch.as_tensor(embedding, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if embedding.dim() != 3 or mask.dim() != 2 or embedding.shape[:2] != mask.shape:
            raise ValueError("Invalid embedding or mask shape in {}".format(path))
        if not torch.isfinite(embedding).all() or (mask.sum(dim=1) == 0).any():
            raise ValueError("Invalid embedding values or empty mask in {}".format(path))
        return embedding, mask


def _first_existing(paths: List[str], description: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find {}. Tried: {}".format(description, paths))


def resolve_split_paths(
    data_folder: str,
    split: str,
    allow_cluster_test_as_val: bool,
) -> Tuple[str, str, str]:
    if split in {"random", "cold"}:
        train_path = _first_existing(
            [os.path.join(data_folder, "processed_train.csv"), os.path.join(data_folder, "train.csv")],
            "training CSV",
        )
        validation_path = _first_existing(
            [os.path.join(data_folder, "processed_val.csv"), os.path.join(data_folder, "val.csv")],
            "validation CSV",
        )
        test_path = _first_existing(
            [os.path.join(data_folder, "processed_test.csv"), os.path.join(data_folder, "test.csv")],
            "test CSV",
        )
    elif split == "cluster":
        train_path = _first_existing(
            [
                os.path.join(data_folder, "processed_source_train.csv"),
                os.path.join(data_folder, "source_train.csv"),
                os.path.join(data_folder, "processed_train.csv"),
            ],
            "cluster training CSV",
        )
        test_path = _first_existing(
            [
                os.path.join(data_folder, "processed_target_test.csv"),
                os.path.join(data_folder, "target_test.csv"),
                os.path.join(data_folder, "processed_test.csv"),
            ],
            "cluster test CSV",
        )
        candidates = [
            os.path.join(data_folder, "processed_target_val.csv"),
            os.path.join(data_folder, "target_val.csv"),
            os.path.join(data_folder, "processed_val.csv"),
            os.path.join(data_folder, "processed_source_val.csv"),
            os.path.join(data_folder, "source_val.csv"),
        ]
        if allow_cluster_test_as_val:
            candidates.append(test_path)
        validation_path = _first_existing(candidates, "cluster validation CSV")
        if os.path.abspath(validation_path) == os.path.abspath(test_path) and not allow_cluster_test_as_val:
            raise ValueError("Validation and test CSVs are identical.")
    else:
        raise ValueError("Unsupported split: {}".format(split))
    return train_path, validation_path, test_path


def create_dataloaders(args):
    data_folder = os.path.join(args.data_root, args.dataset, args.split)
    paths = resolve_split_paths(data_folder, args.split, args.allow_cluster_test_as_val)
    datasets = [
        DTIDataset(
            pd.read_csv(path),
            args.dataset,
            args.drug_view,
            args.protein_view,
            args.embedding_root,
        )
        for path in paths
    ]
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_pairs,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    return (
        DataLoader(datasets[0], shuffle=True, drop_last=True, **common),
        DataLoader(datasets[1], shuffle=False, drop_last=False, **common),
        DataLoader(datasets[2], shuffle=False, drop_last=False, **common),
    )
