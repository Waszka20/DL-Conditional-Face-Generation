# Conditional Face Generation using CVAE

This project explores conditional face generation with a Conditional Variational Autoencoder (CVAE) trained on CelebA. The model learns to generate and edit face images based on a vector of binary facial attributes.

## What the project does

- Generates new faces from a latent vector and attribute set
- Reconstructs CelebA faces from the encoder-decoder pipeline
- Edits selected attributes such as `Male`, `Eyeglasses`, `Blond_Hair`, and `Smiling`
- Compares different beta values to show the reconstruction vs. generation trade-off

## Main files

- `main.py` - experiment entry point, training, generation, and reconstruction export
- `src/model.py` - CVAE and the alternative fully connected CVAE2 model
- `src/train.py` - training loop, beta warm-up, and evaluation helpers
- `src/utils.py` - generation, reconstruction, editing, and interpolation utilities
- `projekt.ipynb` - results, experiments, analysis, and figures used in the presentation

## Dataset

The project uses the CelebA dataset with 40 binary attributes. Images are loaded through `src/dataset.py`, resized for training, and normalized to the `[-1, 1]` range.

## Current setup

The current configuration is stored in `config.yaml` and controls:

- latent size
- image resolution
- beta value
- batch size
- number of epochs
- training subset size

`main.py` reads the config file, trains the model, saves the checkpoint, and exports generated samples and reconstructions to `experiments/`.

## Experiment outputs

The repository already contains saved experiment folders with:

- model checkpoints
- training histories
- reconstruction figures
- generated samples

These artifacts are used by the notebook and presentation slides.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training and export the latest experiment outputs:

```bash
python main.py
```

The run will create a folder under `experiments/` with the current config name and save:

- `model.pth`
- `history.json`
- generated face grids
- reconstruction plots

## Notebook

`projekt.ipynb` is the analysis notebook. It collects:

- reconstruction examples
- attribute interpolation results
- attribute editing examples
- latent-space experiments
- beta trade-off observations

## Limitations

- Some attributes are still weakly controlled
- Fine facial details are blurred
- Rare attribute combinations are harder to generate


