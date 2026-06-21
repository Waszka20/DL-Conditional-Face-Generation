import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def build_experiment_name(cfg):

    return (
        f"cvae"
        f"_z{cfg['model']['latent_dim']}"
        f"_beta{cfg['training']['beta']}"
        f"_lr{cfg['training']['lr']}"
        f"_bs{cfg['training']['batch_size']}"
        f"_epochs{cfg['training']['epochs']}"
        f"_trainLen{cfg['dataset']['train_subset_size']}"
        f"_attrs{len(cfg['attributes'])}"
    )