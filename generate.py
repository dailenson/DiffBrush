# Phase-0 patch: single-process rewrite of upstream generate.py.
# Upstream required `torchrun --nproc_per_node=4` + nccl + multi-stage FID/HWD loops.
# Device table (M2 Max / MPS) and the German line transfer need one process:
#   - `--device` places everything (cuda|mps|cpu), no DistributedDataParallel.
#   - default mode = smoke probe (report-friendly): bundled test_data styles × texts
#     from probe_texts.txt (--text-file; repeatable --text overrides), incl. German
#     umlaut texts through a letters-extended probe dataset (Phase-2 charset, offline).
#   - `--full_eval` reproduces the upstream two-loop protocol (hwd/fid dirs,
#     wikitext-103 35..61-char lines, 5000-image cap), writer chunking preserved.
# Model/data/VAE semantics untouched vs upstream generate.py.
"""Single-process DiffBrush sampler (Phase-0 rewrite of the upstream generate.py)."""
import os
import sys
import argparse
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # host-machine config (.env); CLI flags and real env vars take precedence
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from models.unet import UNetModel
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from tqdm import tqdm
from utils.util import fix_seed
from data_loader.base_dataset import GenerateDataset
from data_loader.IAMDataset import letters, fixed_len

WRITER_NUMS = 496

# Phase-0 / Phase-2 charset: IAM letters + the German extras. Appending (not inserting)
# keeps every existing index unchanged, so English glyphs and the content CNN stay
# weights-neutral. The 7 extras are the Phase-0 subset; Phase 2 later appends the
# remaining audited characters on top of this same table.
GERMAN_CHARS = "äöüßÄÖÜ"
ALLOWED_CHARS = set(letters + GERMAN_CHARS)


class ProbeGenerateDataset(GenerateDataset):
    """GenerateDataset with German-capable letters; identical mechanics otherwise.

    `get_content` looks up a 16×16 Unifont bitmap per character and feeds those to
    the content CNN — there is no embedding table. Unknown characters have no
    bitmap, so they must not appear in probe text (validated in `load_texts`).
    """

    def __init__(self, style_path, type, ref_num, content_type="unifont"):
        configs = {
            "style_path": style_path,
            "type": type,
            "content_type": content_type,
            "fixed_len": fixed_len,
            "letters": letters + GERMAN_CHARS,
            "ref_num": ref_num,
        }
        super().__init__(configs)


def load_texts(args):
    """Return the smoke-probe sentences.

    Repeatable `--text` wins; otherwise the non-comment lines of `--text-file`
    (default `probe_texts.txt`). Every character must exist in `ALLOWED_CHARS`
    or `get_content` would KeyError at sample time.
    """
    if args.text:
        texts = args.text
    else:
        path = Path(args.text_file)
        if not path.exists():
            sys.exit(f"probe text file not found: {path} (create it or pass --text)")
        texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        texts = [t for t in texts if t and not t.startswith("#")]
        if not texts:
            sys.exit(f"no probe texts found in {path}")
    for t in texts:
        bad = sorted({c for c in t if c not in ALLOWED_CHARS})
        if bad:
            sys.exit(
                f"probe text has chars with no glyph bitmap: {bad!r} in {t!r}\n"
                f"allowed: IAM letters + {GERMAN_CHARS!r}"
            )
    return texts


def load_models(args, device):
    """Load config, UNet checkpoint, and frozen SD-1.5 VAE onto `device`."""
    """load config file into cfg"""
    cfg_from_file(args.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """build model architecture"""
    diffusion = Diffusion(device=device)
    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM,
        nb_classes=WRITER_NUMS,
    ).to(device)

    """load pretrained model"""
    unet.load_state_dict(torch.load(args.pretrained_model, map_location="cpu"))
    print(f"load pretrained model from {args.pretrained_model}")
    unet.eval()

    """Load and Freeze VAE Encoder"""
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    return unet, vae, diffusion


def sample_lines(args, unet, vae, diffusion, device, texts, out_dir, subdir="hwd",
                 fid_cap=None):
    """Sample one line per test-set writer for each text and write PNGs.

    Each DataLoader item is one style-reference batch covering every test
    writer (chunked at 65, same as upstream, to bound peak memory). Two
    layouts, matching the upstream two-loop protocol:

    - `subdir="hwd"` (default / smoke): `out_dir/hwd/<wid>/*.png`
    - `subdir="fid"` (`--full_eval` second loop): flat `out_dir/fid/*.png`,
      stop at `fid_cap` (upstream 5000).
    """
    dataset = ProbeGenerateDataset(cfg.TEST.STYLE_PATH, "test", len(texts))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    loader_iter = iter(loader)
    writer_num_each_time = 65
    times = []
    flat_save = subdir == "fid"
    if flat_save and fid_cap:
        fid_dir = os.path.join(out_dir, "fid")
        os.makedirs(fid_dir, exist_ok=True)
        fid_count = len(os.listdir(fid_dir))
    else:
        fid_count = 0
    with torch.no_grad():
        for x_text in tqdm(texts, position=0, desc="text"):
            if flat_save and fid_cap and fid_count >= fid_cap:
                break
            """one style-ref batch per text (all test writers)"""
            data = next(loader_iter)
            style_ref, wid = data["style"][0], data["wid"]
            style_idx = data["style_idx"]
            chunks = []
            for i in range(0, style_ref.shape[0], writer_num_each_time):
                chunks.append((style_ref[i:i + writer_num_each_time],
                               wid[i:i + writer_num_each_time],
                               style_idx[i:i + writer_num_each_time]))
            for style_input, wid_chunk, style_idx_chunk in chunks:
                style_input = style_input.to(device)
                """glyph-CNN content: Unifont bitmaps for this text, repeated per writer"""
                text_ref = dataset.get_content(x_text)
                text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)
                """latent canvas is always 4×8×128 (VAE of 64×1024), independent of text length"""
                x = torch.randn(
                    (text_ref.shape[0], 4, style_input.shape[2] // 8,
                     dataset.fixed_len // 8),
                    device=device,
                )
                t0 = time.time()
                images = diffusion.ddim_sample(
                    unet, vae, style_input.shape[0], x, style_input, text_ref,
                    args.sampling_timesteps, args.eta,
                )
                dt = time.time() - t0
                times.append(dt)
                print(f"  '{x_text[:28]}' x{len(wid_chunk)} writers: {dt:.1f}s")
                stem = "".join(c if c.isalnum() else "_" for c in x_text)[:24]
                for index in range(len(images)):
                    im = torchvision.transforms.ToPILImage()(images[index]).convert("L")
                    name = f"{wid_chunk[index][0]}-{stem}-{style_idx_chunk[index][0]}.png"
                    if flat_save:
                        base, folder = out_dir, "fid"
                    else:
                        base, folder = out_dir, os.path.join(subdir, wid_chunk[index][0])
                    path = os.path.join(base, folder)
                    os.makedirs(path, exist_ok=True)
                    im.save(os.path.join(path, name))
                    fid_count += 1 if flat_save else 0
    if times:
        print(f"sampling done: {sum(times):.0f}s total, {sum(times)/len(times):.1f}s/text-batch")
    return times


def full_eval(args, unet, vae, diffusion, device):
    """--full_eval: upstream's two-loop protocol (hwd per writer, then capped fid dir)."""
    """Generate HWD (one line per writer, all qualifying corpus texts)"""
    with open("data/wikitext103.te") as _f:
        corpus = [l.strip() for l in _f]
    corpus = [t for t in corpus if 35 <= len(t) <= 61]
    sample_lines(args, unet, vae, diffusion, device, corpus, args.save_dir,
                 subdir="hwd")
    """Generate FID (flat dir, 5000-image cap; corpus shuffled like upstream)"""
    random.shuffle(corpus)
    sample_lines(args, unet, vae, diffusion, device, corpus, args.save_dir,
                 subdir="fid", fid_cap=5000)


def main():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", default="configs/IAM.yml",
                        help="Config file (IAM.yml: EMB_DIM, test style path)")
    parser.add_argument("--dir", dest="save_dir",
                        default=os.environ.get("SAVE_DIR", "reports/phase0_probe"),
                        help="Output dir for generated line PNGs")
    parser.add_argument("--pretrained_model", dest="pretrained_model",
                        default=os.environ.get("PRETRAINED_MODEL",
                                               "model_zoo/checkpoint/diffbrush_iam.pt"),
                        help="IAM checkpoint to sample from")
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "mps"),
                        help="cuda | mps | cpu")
    parser.add_argument("--stable_dif_path", type=str,
                        default=os.environ.get("STABLE_DIF_PATH",
                                               "model_zoo/stable-diffusion-v1-5"),
                        help="Frozen SD-1.5 dir (vae/ subfolder)")
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--full_eval", action="store_true",
                        help="upstream two-loop FID/HWD protocol instead of smoke probe")
    parser.add_argument("--text-file", dest="text_file", default="probe_texts.txt",
                        help="probe texts, one per line (# comments ignored)")
    parser.add_argument("--text", action="append", default=[],
                        help="probe text (repeatable; overrides --text-file)")
    args = parser.parse_args()
    device = torch.device(args.device)

    unet, vae, diffusion = load_models(args, device)

    """Generate: smoke probe (default) or upstream FID/HWD (--full_eval)"""
    if args.full_eval:
        full_eval(args, unet, vae, diffusion, device)
    else:
        sample_lines(args, unet, vae, diffusion, device, load_texts(args), args.save_dir)


if __name__ == "__main__":
    main()
