# MEIM – Multi-partition Embedding Interaction iMproved
## Knowledge Graph Link Prediction in Nim

Implementation of the paper:

> Hung-Nghiep Tran, Atsuhiro Takasu.  
> **MEIM: Multi-partition Embedding Interaction Beyond Block Term Format for Efficient and Expressive Link Prediction.**  
> IJCAI-22, 2022. [[paper]](https://github.com/tranhungnghiep/MEIM)

---

## Project layout

```
meim/
├── meim.nimble            # Nimble package descriptor
├── src/
│   ├── tensor.nim         # Pure-Nim dense tensor / matrix library
│   ├── kg_loader.nim      # Knowledge-graph loader (CSV / TSV / NT, auto-detect)
│   ├── meim_model.nim     # MEIM model: score function, soft-ortho loss, Adam
│   ├── trainer.nim        # Training loop, 1-vs-all / k-vs-all CE, evaluation
│   └── meim.nim           # CLI entry point
└── example_data/
    ├── train.csv
    ├── valid.csv
    └── test.csv
```

---

## Build

Requires **Nim ≥ 1.6**. No external packages are needed (stdlib only).

```bash
nim c -d:release -o:meim src/meim.nim
```

---

## Quick start

### Built-in demo (no data required)
```bash
./meim demo
```
Trains on a 29-triple toy KG of scientists and their nationalities/fields,
then predicts top-5 tails/heads and prints individual triple scores.

### Train on your own data
```bash
./meim train --data /path/to/kg_directory
```
The directory must contain `train.{csv,tsv,txt}` and optionally
`valid.{csv,tsv,txt}` and `test.{csv,tsv,txt}`.

Each line of a file is one triple: `head<delim>tail<delim>relation`.
Delimiter is auto-detected (tab, comma, space). Column order is
`head, tail, relation` by default (0-indexed columns 0, 1, 2).

### Predict
```bash
./meim predict --data /path/to/kg --head albert_einstein --rel field --topk 10
./meim predict --data /path/to/kg --tail physics          --rel field --topk 10
```

---

## Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--K` | 3 | Number of embedding partitions |
| `--Ce` | 100 | Entity embedding dim per partition (total = K×Ce) |
| `--Cr` | 100 | Relation embedding dim per partition (total = K×Cr) |
| `--lr` | 3e-3 | Adam learning rate |
| `--decay` | 1.0 | Multiplicative LR decay per epoch |
| `--epochs` | 1000 | Max training epochs |
| `--batch` | 1024 | Mini-batch size |
| `--ortho` | 0.1 | λ_ortho — soft-orthogonality weight |
| `--unorm` | 5e-4 | λ_unitnorm — unit-norm penalty weight |
| `--idrop` | 0.0 | Input dropout rate |
| `--hdrop` | 0.0 | Hidden-layer dropout rate |
| `--eval` | 10 | Validate every N epochs |
| `--kvsall` | off | Use k-vs-all instead of 1-vs-all sampling |
| `--kk` | 500 | k for k-vs-all |

### Paper-recommended settings

| Dataset | K | Ce | Cr | λ_ortho | λ_unitnorm | sampling |
|---------|---|----|----|---------|------------|----------|
| WN18RR | 3 | 100 | 100 | 1e-1 | 5e-4 | k-vs-all |
| FB15K-237 | 3 | 100 | 100 | 0 | 0 | 1-vs-all |
| YAGO3-10 | 5 | 100 | 100 | 1e-3 | 0 | 1-vs-all |

---

## Architecture

```
Score(h, t, r) = Σ_{k=1}^{K}  h_k^T · M_{W,r,k} · t_k

where  M_{W,r,k} = W_k ×₃ r_k   (Ce×Ce matrix, mode-3 product of
                                   core tensor W_k ∈ R^{Ce×Ce×Cr}
                                   with relation partition r_k ∈ R^{Cr})
```

**Two improvements over MEI:**

1. **Independent core tensors** — each partition k has its own `W_k`,
   promoting diverse local interactions (ensemble boosting).

2. **Soft orthogonality** — adds `λ_ortho · Σ_k ‖M_{W,r,k}^T M_{W,r,k} − I‖_F²`
   to the loss, pushing mapping matrices toward orthogonality (max-rank).
   The strength is tunable per dataset, unlike rigid-orthogonality models.

**Loss:**
```
L = L_CE(1-vs-all or k-vs-all softmax) + L_ortho
```

**Optimiser:** Adam with optional exponential LR decay.

**Evaluation:** Filtered MRR, H@1, H@3, H@10 (Bordes et al., 2013 protocol).

---

## Module descriptions

### `tensor.nim`
A minimal dense-tensor library (no external dependencies).
Provides 1–4-D tensors stored as flat `seq[float32]`, matrix multiply,
transpose, dot product, outer product, row slicing, dropout, batch-norm,
softmax, and log-sum-exp.

### `kg_loader.nim`
Loads a KG from a directory or single file.
- Auto-detects CSV / TSV / space-delimited / N-Triples formats.
- Builds entity and relation string→int vocabularies.
- Provides `buildFilterIndex` for filtered evaluation.

### `meim_model.nim`
Core model logic:
- `initParams` — initialises all learnable parameters.
- `computeMappingMatrix` — implements M_{W,r,k} = W_k ×₃ r_k.
- `scoreTriple / scoreAllTails / scoreAllHeads` — forward pass.
- `accumulateScoreGrad` — analytical back-prop through the score function.
- `computeOrthoLoss / accumulateOrthoGrad` — soft-orthogonality term.
- `Optimiser` — Adam with per-tensor moment buffers.

### `trainer.nim`
- `trainBatch1vsAll / trainBatchKvsAll` — mini-batch training steps.
- `evaluate` — filtered MRR / H@k on any split.
- `train` — full epoch loop with validation and LR decay.
- `predictTails / predictHeads / scoreTripleByName` — inference helpers.

---

## Limitations / future work

- **No model persistence** — parameters are not saved to disk yet.
  Add `std/streams`-based serialisation of the flat `seq[float32]` fields.
- **CPU only** — no GPU backend. For large KGs (YAGO3-10) this is slow;
  consider porting the inner loops to C via Nim's FFI or adding OpenMP pragmas.
- **Dense 1-vs-all** — scoring all |E| entities per triple is O(|E|·K·Ce²·Cr).
  For very large |E| (>100k), k-vs-all (`--kvsall`) is strongly preferred.
