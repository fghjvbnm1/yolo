"""
YOLOv8 All-in-One: Dependency Install + Dataset Download + Train
Usage:
    python yolov8_train.py                        # COCO128 (default, quick test)
    python yolov8_train.py --dataset coco          # Full COCO
    python yolov8_train.py --roboflow <url>        # Roboflow dataset
    python yolov8_train.py --data /path/to/data.yaml  # Local custom dataset
"""

import subprocess
import sys
import importlib.util
from pathlib import Path


# ── Dependency installer (runs before any third-party import) ────────────────

def _run_pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args], stdout=subprocess.DEVNULL)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _detect_torch_cuda_tag() -> str:
    """Pick the right PyTorch wheel based on the nvidia-smi driver version."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        major = int(out.split(".")[0])
        if major >= 525:
            return "cu121"
        if major >= 520:
            return "cu118"
        if major >= 450:
            return "cu116"
    except Exception:
        pass
    return "cpu"


def _install_torch():
    tag = _detect_torch_cuda_tag()
    print(f"[setup] Installing PyTorch ({tag}) ...")
    index = {
        "cu121": "https://download.pytorch.org/whl/cu121",
        "cu118": "https://download.pytorch.org/whl/cu118",
        "cu116": "https://download.pytorch.org/whl/cu116",
        "cpu":   "https://download.pytorch.org/whl/cpu",
    }[tag]
    _run_pip("install", "-q", "torch", "torchvision", "torchaudio",
             "--index-url", index)


def _ensure_libgl():
    """Install libGL.so.1 via apt if missing (needed by opencv on headless servers)."""
    import ctypes.util
    if ctypes.util.find_library("GL"):
        return
    print("[setup] libGL.so.1 not found — installing via apt ...")
    try:
        subprocess.check_call(
            ["apt-get", "install", "-y", "-q", "libgl1", "libglib2.0-0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("[setup] libgl1 installed")
    except Exception:
        print("[setup] apt-get failed. Try manually: apt-get install -y libgl1")


def ensure_dependencies():
    print("[setup] Checking dependencies ...")

    _ensure_libgl()

    if not _installed("torch"):
        _install_torch()
    else:
        print("[setup] torch        already installed")

    # Always ensure headless opencv is installed (avoids libGL.so.1 on headless servers)
    installed_pkgs = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format=columns"],
        text=True,
    ).lower()
    has_headless = "opencv-python-headless" in installed_pkgs
    has_full     = "opencv-python " in installed_pkgs  # trailing space excludes headless

    if has_full:
        print("[setup] Replacing opencv-python with headless variant ...")
        _run_pip("uninstall", "-y", "opencv-python")
        _run_pip("install", "-q", "opencv-python-headless")
    elif not has_headless:
        print("[setup] Installing opencv-python-headless ...")
        _run_pip("install", "-q", "opencv-python-headless")
    else:
        print("[setup] opencv-python-headless       already installed")

    for pkg, module in [
        ("ultralytics",  "ultralytics"),
        ("pillow",       "PIL"),
        ("pyyaml",       "yaml"),
        ("matplotlib",   "matplotlib"),
        ("tqdm",         "tqdm"),
    ]:
        if not _installed(module):
            print(f"[setup] Installing {pkg} ...")
            _run_pip("install", "-q", pkg)
        else:
            print(f"[setup] {pkg:<28} already installed")

    # Print torch + GPU summary
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"[setup] torch {torch.__version__} | "
          f"CUDA {'available: ' + torch.cuda.get_device_name(0) if cuda_ok else 'NOT available (CPU mode)'}")
    print()


# ────────────────────────────────────────────────────────────────────────────

def download_roboflow(url: str, dest: Path) -> Path:
    import urllib.request, zipfile

    zip_path = dest / "dataset.zip"
    print(f"[download] Downloading from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"[download] Extracting to {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()

    yamls = list(dest.rglob("data.yaml"))
    if not yamls:
        raise FileNotFoundError("data.yaml not found in downloaded dataset.")
    return yamls[0]


def get_data_yaml(args, dest: Path) -> str:
    if args.data:
        return args.data
    if args.roboflow:
        return str(download_roboflow(args.roboflow, dest))

    dataset_map = {
        "coco128":    "coco128.yaml",       # 128 images  — quick smoke test
        "coco":       "coco.yaml",           # 118k images — standard benchmark
        "voc":        "VOC.yaml",            # ~17k images — Pascal VOC 07+12
        "objects365": "Objects365.yaml",     # 1.7M images — very heavy
        "openimages": "open-images-v7.yaml", # 1.7M images — very heavy
    }
    if args.dataset not in dataset_map:
        
        raise ValueError(
            f"Unknown dataset '{args.dataset}'. "
            f"Choose from: {list(dataset_map.keys())} or use --roboflow / --data."
        )
    return dataset_map[args.dataset]


def main():
    import argparse

    # ── 1. Install dependencies first ────────────────────────────────────────
    ensure_dependencies()

    # ── 2. Parse args ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="YOLOv8 dataset download + train")

    ds_group = parser.add_mutually_exclusive_group()
    ds_group.add_argument("--dataset", default="voc",
                          choices=["coco128", "coco", "voc", "objects365", "openimages"],
                          help="Built-in ultralytics dataset (default: voc)")
    ds_group.add_argument("--roboflow", metavar="URL",
                          help="Roboflow dataset export URL (zip)")
    ds_group.add_argument("--data", metavar="PATH",
                          help="Path to a local data.yaml")

    # fmt: off
    MODELS = [
        # YOLOv8
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        # YOLOv9
        "yolov9c.pt", "yolov9e.pt",
        # YOLOv10
        "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
        # YOLO11 (ultralytics latest)
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    ]
    # fmt: on
    parser.add_argument("--model",   default="yolov8n.pt", choices=MODELS,
                        help="Model weight file (default: yolov8n)")
    parser.add_argument("--epochs",  type=int, default=500)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--device",  default="0", help="GPU id or 'cpu'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name",    default="yolov8_exp")

    args = parser.parse_args()

    # ── 3. Resolve dataset ───────────────────────────────────────────────────
    from ultralytics import YOLO

    dest = Path("datasets") / "custom"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"{'='*55}")
    print(" YOLOv8 All-in-One Training Script")
    print(f"{'='*55}")

    data_yaml = get_data_yaml(args, dest)
    print(f"[config] dataset  : {data_yaml}")
    print(f"[config] model    : {args.model}")
    print(f"[config] epochs   : {args.epochs}")
    print(f"[config] imgsz    : {args.imgsz}")
    print(f"[config] batch    : {args.batch}")
    print(f"[config] device   : {args.device}")
    print(f"{'='*55}\n")

    # ── 4. Load model (auto-redownload if weights are corrupted) ─────────────
    def load_model(name: str):
        pt = Path(name)
        for attempt in range(2):
            try:
                return YOLO(name)
            except RuntimeError as e:
                if "zip archive" in str(e) and pt.exists():
                    print(f"[warn] {name} is corrupted (attempt {attempt+1}). Deleting and retrying ...")
                    pt.unlink()
                else:
                    raise
        raise RuntimeError(f"Failed to load {name} after re-download.")

    model = load_model(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    # ── 5. Validate best checkpoint ──────────────────────────────────────────
    print("\n[validate] Running validation on best.pt ...")
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        metrics = YOLO(str(best_pt)).val(data=data_yaml, device=args.device)
        print(f"\n[result] mAP50    : {metrics.box.map50:.4f}")
        print(f"[result] mAP50-95 : {metrics.box.map:.4f}")
    else:
        print(f"[warn] best.pt not found at {best_pt}")

    print(f"\n[done] Weights saved to: {Path(args.project) / args.name / 'weights'}")


if __name__ == "__main__":
    main()
