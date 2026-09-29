# CFGB-DTI

PyTorch implementation of **CFGB-DTI**, a cross-foundation graph bottleneck framework for drug–target interaction prediction. The model constructs a drug–protein evidence graph from frozen foundation-model embeddings and learns a variational graph bottleneck bridge for prediction.

## Requirements

- Python 3.8
- PyTorch 1.8.1
- NumPy
- pandas
- scikit-learn
- tqdm

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

Each split CSV must contain `SMILES_id`, `Protein_id`, and `Y`. Random and cold splits use:

```text
<data_root>/<dataset>/<split>/processed_train.csv
<data_root>/<dataset>/<split>/processed_val.csv
<data_root>/<dataset>/<split>/processed_test.csv
```

Cluster splits use source training and target validation/test files. Precomputed embeddings are stored under:

```text
<embedding_root>/<dataset>/smiles_embeddings/
<embedding_root>/<dataset>/unimol_embeddings/
<embedding_root>/<dataset>/protein_embeddings/
<embedding_root>/<dataset>/structure_embeddings/
```

Each `.pt` file contains an embedding tensor and its token mask.

## Run

```bash
python main.py \
  --dataset bindingdb \
  --split random \
  --data_root /path/to/datasets \
  --embedding_root /path/to/embeddings \
  --device cuda:0
```
