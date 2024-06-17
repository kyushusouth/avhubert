from pathlib import Path

import hydra
import omegaconf
import torch

from avhubert import AVHuBERT


def load_avhubert_from_torch_checkpoint(cfg: omegaconf.DictConfig) -> AVHuBERT:
    avhubert = AVHuBERT(cfg)
    ckpt_path = Path(cfg.ckpt_path).expanduser()
    if cfg.load_pretrained_weight:
        pretrained_dict = torch.load(str(ckpt_path))["avhubert"]
        avhubert.load_state_dict(pretrained_dict, strict=True)
    return avhubert


@hydra.main(config_path="./conf", config_name="base")
def main(cfg: omegaconf.DictConfig) -> None:
    avhubert = load_avhubert_from_torch_checkpoint(cfg)


if __name__ == "__main__":
    main()
