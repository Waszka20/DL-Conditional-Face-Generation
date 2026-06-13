## Conditional Face Generation with CVAE (CelebA)

This project explores **conditional generative modeling** using a Conditional Variational Autoencoder (CVAE) trained on the CelebA dataset(https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data). The goal is to generate human face images with controllable attributes such as *smiling*, *eyeglasses*, or *hair color*.

We will try to apply **multi-attribute conditioning**, where the model takes a vector of facial features as input. This would allows us to generate faces with combinations of attributes (e.g., “smiling + glasses”) and experiment with how well the model can control specific features.

### Project goals

* Learn a meaningful **latent space representation** of faces
* Generate images based on selected attributes
* Evaluate generation quality


This project is part of a deep learning course and focuses on gaining hands-on experience with generative models and conditional learning on a more realistic dataset.

---

## Team

* *Piotr Waszak*
* *Filip Bętkowski*

---

## English TODO & Run Instructions

Below is the consolidated TODO list (translated to English) and quick instructions to run experiments.

### Priority — High
- Run the notebook (`projekt.ipynb`) and perform experiments.
- Add evaluation metrics (FID, Inception Score, reconstruction/validation losses on a test set).
- Compare `CVAE` vs `CVAE2` (train both, collect quantitative and qualitative results).
- Analyze the latent space (t-SNE / UMAP visualizations, attribute separation).
- Test and validate attribute editing on real images using `edit_face_attributes()`.

### Priority — Medium
- Add a validation loop and checkpointing in training.
- Expand conditioning to more attributes (e.g., ~50 attributes).
- Perform error analysis to find failure modes (mode collapse, attribute entanglement).
- Improve README with installation, data download, run instructions and example results.

### Priority — Low
- Add docstrings and inline documentation for functions and classes.
- Refactor code and remove or integrate unused `CVAE2` logic.

---

### Quick setup
Install required packages (preferably in a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate    # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

Note: `kagglehub` is used in `src/dataset.py` to download the CelebA dataset; ensure you have network access and any required credentials.

### Run training (example)
Run the provided `main.py` to train a CVAE (adjust `TRAIN` in `main.py` or modify the script to pass args):

```bash
python main.py
```

### Evaluate and visualize
Use the utilities in `src/eval.py` for reconstruction metrics and latent t-SNE visualization. Example usage (python REPL):

```python
from src.eval import compute_latent_tsne
from src.dataset import CelebADataset
from src.model import CVAE

# load data and model (example)
# dataset = CelebADataset(split='eval', SELECTED_ATTRIBUTES=[...])
# loader = DataLoader(dataset, batch_size=64)
# model = CVAE(...)
# model.load_state_dict(torch.load('cvae.pth'))

# compute_latent_tsne(model, loader, attribute_names, device='cuda')
```

### Notes
- Full FID computation requires `pytorch-fid` and an images folder of real examples; `src/eval.py` contains a helper that will raise a helpful message if the dependency is missing.
- Training on CelebA is computationally intensive; run on a GPU and consider using smaller subsets for quick experiments.

If you want, I can now: run the notebook cells, implement a training script that trains both `CVAE` and `CVAE2` and logs metrics, or add checkpointing and validation (I've already added validation+checkpoint support in `src/train.py`). Which should I do next?
