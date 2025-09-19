# Semantic Segmentation for Chinese Architecture

This repository contains a TensorFlow 2 implementation of a Dynamic Graph CNN (DGCNN) with attention aggregation for semantic segmentation of point clouds captured from Chinese architectural heritage.

## Repository Layout

| File | Description |
| ---- | ----------- |
| `config.py` | Central location for hyper-parameters (class colors, data paths, optimizer settings, etc.). |
| `data_utils.py` | Utilities that transform raw `.txt` point clouds into model-ready `.h5` datasets through blocking, sampling, and feature formatting. |
| `provider.py` | Helper functions for loading `.h5` files, handling file paths, and simple augmentations/statistics. |
| `model_Att.py` | Attention-enhanced DGCNN definition and loss function implemented with `tf.keras`. |
| `train.py` | End-to-end training script that prepares datasets, builds the model, and handles logging/checkpointing. |
| `application.py` | Standalone inference and evaluation script for a single room/area using trained weights. |

## Environment Setup

1. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install tensorflow==2.3 numpy h5py
   ```
   *The model was authored against TensorFlow 2.3. Later 2.x releases generally work, but matching the pinned version avoids API incompatibilities.*
3. **(Optional) GPU acceleration:** install the TensorFlow GPU build that matches your CUDA/cuDNN stack to speed up training.

## Data Preparation Pipeline

1. **Organise raw data:**
   - Place your raw room files inside `./data/`. Each file should contain per-point rows with space-separated attributes `X Y Z R G B L` where `L` is the semantic label.
   - Create `./data/rooms.txt` listing one room filename per line. The loader reads this list to know which files to process.
2. **Generate training blocks and datasets:**
   ```bash
   python data_utils.py
   ```
   This command performs the following steps:
   - Reads every room listed in `rooms.txt` and normalises XYZ coordinates so each room starts at the origin.
   - Tiles each room into sliding square blocks, randomly samples a fixed number of points per block, and writes intermediate `*_blocked.h5`/`.txt` files with per-point colour encodings.
   - Converts absolute coordinates and colours to a 9-dimensional feature representation (centred XYZ + normalised XYZRGB) before concatenating all rooms into a single `CABDataset.h5` dataset alongside `rooms_name.txt` containing block identifiers.
3. **Verify outputs:** After the script finishes you should have:
   - `data/CABDataset.h5` containing `data` (B × N × 9 features) and `label` (B × N) arrays.
   - `data/rooms_name.txt` capturing the block names for later train/test splitting.
   - `data/all_files.txt` listing the dataset .h5 files to load during training (e.g., a single line `data/CABDataset.h5`).
   - `<room>_blocked.h5` and `.txt` files in the project root for visual inspection.

## Dataset Format and Customisation

### Expected raw file structure

- The preprocessing scripts assume **plain-text `.txt` point clouds with no header**. Each line must provide seven space-separated columns in the exact order `X Y Z R G B L`.
- Coordinates (`X Y Z`) are treated as metre-scale floats, colours (`R G B`) should be integers in `[0, 255]`, and labels (`L`) must be integer class IDs in `[0, 11]` to align with `config.NUM_CLASSES` and `config.COLOR_DICT`.
- Empty lines or additional columns will cause parsing failures inside `data_utils.txt2Matrix()`. Remove them or update the parser before running `python data_utils.py`.

### Where the format is enforced

- **Parsing:** [`data_utils.txt2Matrix()`](data_utils.py) reads `rooms.txt`, splits every row, and expects the last column to be the label. Adjust this function if your raw files follow a different layout (e.g., extra normals or intensity values).
- **Feature construction:** [`data_utils.formatAttribute()`](data_utils.py) converts raw XYZRGB into the 9-feature tensor stored in `CABDataset.h5`. Extend this function if you introduce extra attributes so that the downstream network receives the correct dimensionality.
- **Runtime loading:** [`provider.getDataFiles()` and `provider.loadH5Files()`](provider.py) look up the generated `.h5` file list in `data/all_files.txt`. Change `config.DATA_PATH`, `config.DATA_FILE`, or these helper functions if you relocate the dataset or alter file names.

### Customising the dataset pipeline

1. Update `config.DATA_FILE` / `config.DATA_PATH` to point at a different room list or directory.
2. Modify `txt2Matrix()` to reinterpret each column if your raw scanner exports extra features; remember to update `formatAttribute()` and any TensorFlow model layers so that input channel counts stay consistent.
3. If you produce multiple `.h5` shards, add every filename to `data/all_files.txt` (or override `provider.getDataFiles()` to glob automatically) before launching `train.py`.
4. Regenerate `CABDataset.h5` after any structural change to ensure training and inference consume the new format.

## Running Logic

1. **Dataset loading:** `train.py` loads all `.h5` files listed in `data/all_files.txt`, concatenates them, and splits rooms into train/test sets by matching the `--test_area` substring against block names read from `rooms_name.txt`.
2. **Sampling weights & augmentation:** Class frequency weights are computed from the training labels and applied during optimisation. Each batch is also randomly rotated around the Z-axis to augment geometric variance.
3. **Model architecture:** `model_Att.py` builds an attention-aware DGCNN. It constructs edge features via KNN lookups, applies stacked attention aggregation layers, pools global context, and predicts per-point logits through 1×1 convolutions.
4. **Training loop:** For every epoch the script iterates over the `tf.data` pipeline, logs metrics to TensorBoard, evaluates on the held-out area at a configurable frequency, and checkpoints the best model weights under `log/`.
5. **Inference:** `application.py` reloads the trained `DGCNN`, selects blocks that belong to `config.TEST_AREA`, runs segmentation, computes OA/mIoU per class, and exports coloured `.txt` files for predictions and ground truth comparison.

## Training

Run the training script after data preparation:
```bash
python train.py --test_area test --log_dir log --epoch 150 --batch_size 4
```
Key flags:
- `--test_area`: substring used to select evaluation blocks (matches entries from `rooms_name.txt`).
- `--learning_rate`, `--decay_steps`, `--decay_rate`: optimiser schedule (defaults defined in `config.py`).
- `--log_dir`: directory for TensorBoard logs and checkpoints.
- `--test_frequency` / `--save_frequency`: how often to evaluate and checkpoint.

Monitor training with TensorBoard:
```bash
tensorboard --logdir log
```

## Evaluation / Inference

To evaluate a trained checkpoint on a specific area:
1. Ensure `config.TEST_AREA` matches the desired room identifier and that the corresponding `*_blocked.h5` file is available.
2. Run:
   ```bash
   python application.py
   ```
   The script reports OA/mIoU, writes per-class metrics to `精度.txt`, and exports `{area}_predicted.txt` / `{area}_labeled.txt` point clouds coloured with `config.COLOR_DICT` for visual inspection.

## Tips

- Adjust hyper-parameters such as `BATCH_SIZE`, `LEARNING_RATE`, or `ZOOM_FACTOR` in `config.py` to tailor experiments.
- When introducing new classes, update `config.NUM_CLASSES` and extend `config.COLOR_DICT` accordingly.
- Keep the raw `.txt` files and generated `.h5` datasets backed up; regenerating them can be time-consuming for large scans.
