from pathlib import Path

import torch

from avhubert import AVHuBERT, Config


def load_avhubert_from_original_checkpoint(
    model_size: str, ckpt_path: Path
) -> AVHuBERT:
    if model_size not in ["base", "large"]:
        raise ValueError("model_size must be 'base' or 'large'")

    cfg = Config(model_size)
    avhubert = AVHuBERT(cfg)
    state = torch.load(ckpt_path, map_location=torch.device("cpu"))
    pretrained_dict = state["model"]
    avhubert_dict = avhubert.state_dict()
    match_dict = {k: v for k, v in pretrained_dict.items() if k in avhubert_dict}
    avhubert.load_state_dict(match_dict, strict=True)
    return avhubert


def main():
    # AVHuBERTのモデルサイズ。baseかlargeのどちらかでお願いします。
    model_size = "large"

    # ダウンロードしたチェックポイントまでのパス
    ckpt_path = "./large_vox_iter5.pt"

    avhubert = load_avhubert_from_original_checkpoint(model_size, ckpt_path)

    # PyTorchのチェックポイントとして保存し直す
    ckpt_path_new = "./large_vox_iter5_torch.ckpt"
    torch.save(
        {
            "avhubert": avhubert.state_dict(),
        },
        ckpt_path_new,
    )


if __name__ == "__main__":
    main()
