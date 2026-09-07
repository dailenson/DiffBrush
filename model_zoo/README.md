# model_zoo

Pre-trained model artifacts (the data subfolders are git-ignored; this README is not). Stage them on a fresh clone with:

```
uv run --with gdown scripts/download_model_zoo.py
```

Expected structure:

```
model_zoo/
├── README.md
├── checkpoint/
│   └── diffbrush_iam.pt              # ~1.1 GB
│                                       #   released IAM checkpoint (upstream "Model Zoo",
│                                       #   Google Drive): slim UNet + Mix_TR + 496-writer
│                                       #   proxy tables (654 tensors, 163.05 M params)
└── stable-diffusion-v1-5/            # frozen SD-1.5 VAE (Hugging Face, upstream
    ├── vae/                          #   component — never trained)
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors   # ~335 MB, fp32
    │                                           #   (the .bin/.fp16 variants are not needed)
    └── scheduler/                    # optional + unused: DiffBrush implements its own
        └── scheduler_config.json     #   DDIM schedule (models/diffusion.py); the
                                      #   download script does not fetch this folder
```

Consumers:

| artifact                      | used as                                                                                                                        |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `checkpoint/diffbrush_iam.pt` | `generate.py --pretrained_model` / `.env` `PRETRAINED_MODEL`; warm start for Phases 3–5                                        |
| `stable-diffusion-v1-5/`      | `generate.py --stable_dif_path` / `.env` `STABLE_DIF_PATH` (loaded with `AutoencoderKL.from_pretrained(..., subfolder="vae")`) |

What the two artifacts are and in which phases they are needed:
see `fine-tuning-plan.md`, Phase 0 → "What the two staged artifacts are, and where they are needed".
