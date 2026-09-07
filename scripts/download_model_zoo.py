"""Download the model_zoo artifacts for a fresh clone.

Stages exactly what the plan's Phase 0 and later phases need:

- model_zoo/checkpoint/
    diffbrush_iam.pt                  released IAM checkpoint (upstream "Model Zoo",
                                       Google Drive, ~1.1 GB)
- model_zoo/stable-diffusion-v1-5/    frozen SD-1.5 VAE (Hugging Face,
  vae/config.json                     https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5,
  vae/diffusion_pytorch_model.safetensors
                                       ~335 MB total; the same subfolder layout
                                       `AutoencoderKL.from_pretrained(..., subfolder="vae")`
                                       expects)

Idempotent: existing files are skipped unless --force. The checkpoint is verified
after download (torch.load, 654 tensors — see reports/phase0_checkpoint.md).

Usage:
    uv run --with gdown scripts/download_model_zoo.py [--force] [--only ckpt|vae]

Notes:
- gdown is only needed for the Drive download; it is pulled on the fly by
  `uv run --with gdown` (not part of the project dependencies).
- If your network requires Hugging Face authentication, `export HF_TOKEN=...`
  first — huggingface_hub picks it up automatically.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_ZOO = REPO / "model_zoo"
DRIVE_URL = "https://drive.google.com/file/d/1EWzBmLtnQ42cTf3k_CYQ-nF3RXCb35I6/view?usp=drive_link"
HF_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_SUBDIR = "stable-diffusion-v1-5/vae"
VAE_FILES = ["config.json", "diffusion_pytorch_model.safetensors"]
EXPECTED_CKPT_TENSORS = 654


def fetch_ckpt(force: bool) -> None:
    out = MODEL_ZOO / "checkpoint" / "diffbrush_iam.pt"
    if out.exists() and not force:
        print(f"skip: {out} exists (--force to re-download)")
        return
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required for the Drive download — run: "
                 "uv run --with gdown scripts/download_model_zoo.py")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.part")
    print(f"downloading checkpoint -> {out}")
    gdown.download(DRIVE_URL, output=str(tmp), fuzzy=True, quiet=False)
    if not tmp.exists() or tmp.stat().st_size < 100_000_000:
        tmp.unlink(missing_ok=True)
        sys.exit("download produced no/little data — is the Drive link reachable?")
    tmp.replace(out)
    import torch
    sd = torch.load(out, map_location="cpu")
    print(f"checkpoint OK: {len(sd)} tensors (expect {EXPECTED_CKPT_TENSORS})")
    if len(sd) != EXPECTED_CKPT_TENSORS:
        print("WARNING: tensor count differs from the released checkpoint — "
              "re-check the Drive link", file=sys.stderr)


def fetch_vae(force: bool) -> None:
    vae_dir = MODEL_ZOO / VAE_SUBDIR
    missing = [f for f in VAE_FILES if force or not (vae_dir / f).exists()]
    if not missing:
        print(f"skip: {vae_dir} complete (--force to re-download)")
        return
    from huggingface_hub import hf_hub_download
    vae_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        print(f"downloading {HF_REPO} {VAE_SUBDIR}/{name}")
        hf_hub_download(HF_REPO, f"vae/{name}", local_dir=str(MODEL_ZOO),
                        force_download=force)
    for name in VAE_FILES:
        p = vae_dir / name
        if not p.exists() or p.stat().st_size == 0:
            sys.exit(f"verification failed: {p} missing or empty")
    print(f"VAE OK: {vae_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--only", choices=["all", "ckpt", "vae"], default="all")
    args = ap.parse_args()
    if args.only in ("all", "ckpt"):
        fetch_ckpt(args.force)
    if args.only in ("all", "vae"):
        fetch_vae(args.force)
    print("done — model_zoo/ is staged (see fine-tuning-plan.md Phase 0 for the re-run command)")


if __name__ == "__main__":
    main()
