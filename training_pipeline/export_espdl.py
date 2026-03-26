"""
export_espdl.py — Export a trained LetterCNN checkpoint to an ESP-DL .espdl model.

Requires:
    pip install esp-ppq

Usage:
    python export_espdl.py                          # uses defaults below
    python export_espdl.py --model best_model.pth --output lettercnn.espdl

The script:
  1. Loads the saved .pth checkpoint.
  2. Builds a calibration dataloader from the same image folder used for training.
  3. Runs esp-ppq post-training quantisation targeting the ESP32-S3 (INT8).
  4. Writes a .espdl file ready to flash alongside your firmware.

Note: the calibration data must use the same preprocessing as training:
  - Resize to 32×32, preserve aspect ratio with white padding
  - RGB (3 channels)
  - Normalise: mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)  →  [-1, 1]
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from esp_ppq.api import espdl_quantize_torch
from esp_ppq.executor.torch import TorchExecutor

# ── re-use definitions from train.py ─────────────────────────────────────────
from train import LetterCNN, SquarePad, InsectImageFolder, VALID_CLASSES, _MEAN, _STD

# ─────────────────────────────────────────────────────────────────────────────

TARGET      = 32
DEVICE      = "cpu"
TARGET_CHIP = "esp32s3"
NUM_BITS    = 8
CALIB_STEPS = 32   # number of calibration batches


def build_calib_loader(data_dir: str, batch_size: int = 32) -> DataLoader:
    """Return a non-shuffled dataloader over the image folder for calibration."""
    tf = transforms.Compose([
        SquarePad(),
        transforms.Resize((TARGET, TARGET)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    ds = InsectImageFolder(data_dir, transform=tf)
    ds.targets = [s[1] for s in ds.samples]
    # shuffle=False is required by esp-ppq: the calibration set is traversed
    # multiple times, and shuffling would give incorrect quantisation error stats.
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def collate_fn(batch):
    """esp-ppq expects only the input tensor, not the (input, label) tuple."""
    return batch[0].to(DEVICE)


def load_model(model_path: str) -> tuple[LetterCNN, dict, dict]:
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    num_classes = len(ckpt["class_to_idx"])
    model = LetterCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["class_to_idx"], ckpt["idx_to_class"]


def evaluate(model_or_executor, loader: DataLoader, is_executor: bool = False) -> float:
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        if is_executor:
            logits = model_or_executor(imgs)[0]
        else:
            with torch.no_grad():
                logits = model_or_executor(imgs)
        total_loss    += criterion(logits, labels).item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += imgs.size(0)
    return total_loss / total, total_correct / total * 100.0


# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Export LetterCNN to ESP-DL .espdl format")
    parser.add_argument("--model",  default=os.path.join(here, "best_model.pth"),
                        help="Path to the .pth checkpoint (default: %(default)s)")
    parser.add_argument("--output", default=os.path.join(here, "lettercnn.espdl"),
                        help="Output .espdl file path (default: %(default)s)")
    parser.add_argument("--data",   default=here,
                        help="Image folder root used for calibration (default: %(default)s)")
    parser.add_argument("--calib-steps", type=int, default=CALIB_STEPS,
                        help="Number of calibration batches (default: %(default)s)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.model}")
    model, class_to_idx, idx_to_class = load_model(args.model)
    print(f"  classes: {class_to_idx}")

    print(f"Building calibration dataloader from: {args.data}")
    calib_loader = build_calib_loader(args.data)
    print(f"  {len(calib_loader.dataset)} calibration samples")

    # Input shape for a single sample: [C, H, W] — batch dimension excluded.
    input_shape = [1, 3, TARGET, TARGET]

    print(f"\nQuantising for {TARGET_CHIP} ({NUM_BITS}-bit) …")
    quant_graph = espdl_quantize_torch(
        model              = model,
        espdl_export_file  = args.output,
        calib_dataloader   = calib_loader,
        calib_steps        = args.calib_steps,
        input_shape        = input_shape,
        inputs             = None,
        target             = TARGET_CHIP,
        num_of_bits        = NUM_BITS,
        collate_fn         = collate_fn,
        dispatching_override = None,
        device             = DEVICE,
        error_report       = True,
        skip_export        = False,
        export_test_values = True,
        verbose            = 1,
    )
    print(f"\nExported: {args.output}")

    # ── accuracy comparison ────────────────────────────────────────────────
    print("\nEvaluating on calibration set …")
    fp_loss, fp_acc = evaluate(model, calib_loader, is_executor=False)
    print(f"  float32 model : loss={fp_loss:.4f}  acc={fp_acc:.1f}%")

    executor = TorchExecutor(graph=quant_graph, device=DEVICE)
    q_loss, q_acc = evaluate(executor, calib_loader, is_executor=True)
    print(f"  INT8 quantised: loss={q_loss:.4f}  acc={q_acc:.1f}%")
    print(f"  accuracy delta: {abs(fp_acc - q_acc):.2f} pp")


if __name__ == "__main__":
    main()
