# Evaluates the seed-11 models from their checkpoints and runs the interventions on the lesion branch; fills runs/seed11_locked_eval_interventions.
import os

STORAGE = os.environ.get("MSLS_STORAGE", "storage")
SCRATCH = os.environ.get("MSLS_SCRATCH", "scratch")


def display(obj):
    print(obj)


import hashlib
import gc
import itertools
import platform
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
import random
import shutil
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode, ColorJitter

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from huggingface_hub import snapshot_download, HfApi
from transformers import AutoModelForImageClassification

import matplotlib.pyplot as plt
import seaborn as sns
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

warnings.filterwarnings("ignore")

CONFIG = {
    "protocol": "clean_protocol_v1",
    "seed": 11,
    "epochs": 40,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "lambda_mask": 0.50,
    "mask_bce_weight": 0.50,
    "mask_dice_weight": 0.50,
    "mask_pos_weight_cap": 20.0,
    "num_workers": 4,
    "skip_completed": True,
    "manifest_root": STORAGE + "/msls_vmamba/data/clean_protocol_v1",
    "plantseg_zip": STORAGE + "/plantseg.zip",
    "plantdoc_root": STORAGE + "/PlantDoc-Dataset",
    "class_mapping": STORAGE + "/msls_vmamba/data/class_mapping.json",
    "drive_runs_root": STORAGE + "/msls_vmamba/runs/seed11_analyses/locked_eval_seed11_v1",
    "local_cache_root": SCRATCH + "/vmamba_clean_protocol_v1",
    "model_repo": "saurabhati/VMamba_ImageNet_82.6",
    "model_dir": SCRATCH + "/vmamba_msls_model",
    "existing_model_dir": SCRATCH + "/vmamba_model",
    "model_revision": None,
}

RUN_SPECS = [
    {
        "name": "vmamba_baseline_strong_384",
        "model": "vmamba_baseline",
        "img_size": 384,
        "batch_size": 16,
        "accumulation_steps": 2,
        "label_smoothing": 0.10,
        "warmup_epochs": 4,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
    },
    {
        "name": "current_full_strong_384",
        "model": "current_full",
        "img_size": 384,
        "batch_size": 16,
        "accumulation_steps": 2,
        "label_smoothing": 0.10,
        "warmup_epochs": 4,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
    },
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "A CUDA GPU is required."
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


NEW_SEEDS = (11, 22, 33, 44, 55)
EXPECTED_PROTOCOL_FINGERPRINT = "ff6e6b3bca40e614062e611a50d6736690a3658d433593f500a0c34c651ec648"
EXTENSION_VERSION = "vmamba_msls_11_55_v1"
EXPECTED_PARAMETERS = {"vmamba_baseline": 29492368, "current_full": 30405907}
assert len(set(NEW_SEEDS)) == 5
set_seed(CONFIG["seed"])
Path(CONFIG["drive_runs_root"]).mkdir(parents=True, exist_ok=True)

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VMamba CUDA kernels: OK")
display(pd.DataFrame(RUN_SPECS))


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

RUNTIME_INFO = {
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "packages": {name: package_version(name) for name in (
        "torch", "torchvision", "numpy", "transformers", "mamba-ssm",
        "causal-conv1d", "huggingface-hub", "scikit-learn", "scipy", "pandas", "timm",
    )},
}
for name, expected in {"torch": "2.2.2", "torchvision": "0.17.2",
                       "transformers": "4.38.2", "mamba-ssm": "2.2.2",
                       "causal-conv1d": "1.2.0.post2"}.items():
    actual = RUNTIME_INFO["packages"][name]
    assert actual and actual.split("+")[0] == expected, (
        f"{name}: expected {expected}, got {actual}."
    )
assert int(np.__version__.split(".")[0]) < 2, "NumPy < 2 is required."

print("Seeds:", NEW_SEEDS)
print("10 training runs.")
print("New results:", CONFIG["drive_runs_root"])

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def load_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)

def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
    }

def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])

EXTENSION_ROOT = Path(CONFIG['drive_runs_root'])
ANALYSIS_ROOT = EXTENSION_ROOT / 'analysis'
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

SOURCE_REFERENCE = {
}


MANIFEST_FILES = {
    "train": "plantseg_optimization_train.csv",
    "validation": "plantseg_optimization_validation.csv",
    "plantseg_test": "plantseg_official_test.csv",
    "plantdoc_external": "plantdoc_external_all.csv",
}
EXPECTED_COUNTS = {
    "train": 1398,
    "validation": 350,
    "plantseg_test": 435,
    "plantdoc_external": 1379,
}

manifest_root = Path(CONFIG["manifest_root"])
plantseg_zip = Path(CONFIG["plantseg_zip"])
plantdoc_root = Path(CONFIG["plantdoc_root"])
mapping_path = Path(CONFIG["class_mapping"])

for required in (manifest_root, plantseg_zip, plantdoc_root, mapping_path):
    assert required.exists(), f"Missing required input: {required}"

with mapping_path.open(encoding="utf-8") as handle:
    mapping = json.load(handle)

CLASS_NAMES = list(mapping["class_names"])
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)
assert NUM_CLASSES == 16

manifests = {}
for role, filename in MANIFEST_FILES.items():
    frame = pd.read_csv(manifest_root / filename)
    assert len(frame) == EXPECTED_COUNTS[role], (
        role, len(frame), EXPECTED_COUNTS[role]
    )
    assert frame["record_id"].is_unique, f"Duplicate record IDs in {filename}"
    assert set(frame["harmonized_class"]) == set(CLASS_NAMES), filename
    frame["label"] = frame["harmonized_class"].map(CLASS_TO_IDX)
    assert frame["label"].notna().all(), filename
    frame["label"] = frame["label"].astype(int)
    manifests[role] = frame.reset_index(drop=True)

train_ids = set(manifests["train"]["record_id"])
validation_ids = set(manifests["validation"]["record_id"])
test_ids = set(manifests["plantseg_test"]["record_id"])
external_ids = set(manifests["plantdoc_external"]["record_id"])

assert train_ids.isdisjoint(validation_ids)
assert train_ids.isdisjoint(test_ids)
assert validation_ids.isdisjoint(test_ids)
assert (train_ids | validation_ids | test_ids).isdisjoint(external_ids)

train_clusters = set(manifests["train"]["duplicate_cluster"])
validation_clusters = set(manifests["validation"]["duplicate_cluster"])
test_clusters = set(manifests["plantseg_test"]["duplicate_cluster"])
external_clusters = set(manifests["plantdoc_external"]["duplicate_cluster"])
assert train_clusters.isdisjoint(validation_clusters)
assert train_clusters.isdisjoint(test_clusters)
assert validation_clusters.isdisjoint(test_clusters)
assert (train_clusters | validation_clusters | test_clusters).isdisjoint(
    external_clusters
)

train_hashes = set(manifests["train"]["sha256"])
validation_hashes = set(manifests["validation"]["sha256"])
test_hashes = set(manifests["plantseg_test"]["sha256"])
external_hashes = set(manifests["plantdoc_external"]["sha256"])
assert train_hashes.isdisjoint(validation_hashes)
assert train_hashes.isdisjoint(test_hashes)
assert validation_hashes.isdisjoint(test_hashes)
assert (train_hashes | validation_hashes | test_hashes).isdisjoint(
    external_hashes
)

fingerprint_roles = dict(manifests)
# The cleaning protocol also recorded PlantDoc's own test split; it is rebuilt here from the PlantDoc manifest so the recorded fingerprint can be verified.
fingerprint_roles["plantdoc_official_test"] = manifests["plantdoc_external"][
    manifests["plantdoc_external"]["source_split"] == "test"
]
fingerprint_payload = []
for role in sorted(fingerprint_roles):
    for row in fingerprint_roles[role].sort_values("record_id").itertuples(index=False):
        fingerprint_payload.append(f"{role}|{row.record_id}|{row.sha256}")
PROTOCOL_FINGERPRINT = hashlib.sha256(
    "\n".join(fingerprint_payload).encode("utf-8")
).hexdigest()

print("Protocol:", CONFIG["protocol"])
print("Fingerprint:", PROTOCOL_FINGERPRINT)
print("All four manifests and dataset boundaries passed.")

count_rows = []
for role, frame in manifests.items():
    counts = frame.groupby("harmonized_class").size()
    for class_name in CLASS_NAMES:
        count_rows.append({
            "manifest": role,
            "class": class_name,
            "images": int(counts.get(class_name, 0)),
        })
display(pd.DataFrame(count_rows).pivot(
    index="class", columns="manifest", values="images"
).astype(int))

# The manifests must match the recorded cleaning protocol before anything is evaluated.
assert PROTOCOL_FINGERPRINT == EXPECTED_PROTOCOL_FINGERPRINT, (
    'These manifests differ from the recorded clean protocol. Stop and inspect the paths.'
)

local_root = Path(CONFIG["local_cache_root"])
local_ps_root = local_root / "plantseg"
local_pd_root = local_root / "plantdoc"
local_ps_root.mkdir(parents=True, exist_ok=True)
local_pd_root.mkdir(parents=True, exist_ok=True)


def sha256_path(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


all_rows = pd.concat(manifests.values(), ignore_index=True)
all_rows = all_rows.drop_duplicates("record_id").reset_index(drop=True)
local_paths = {}
source_mask_hashes = {}

with zipfile.ZipFile(plantseg_zip) as archive:
    archive_members = set(archive.namelist())
    for number, row in enumerate(all_rows.itertuples(index=False), start=1):
        if row.dataset == "PlantSeg":
            assert row.image_reference in archive_members, row.image_reference
            assert row.mask_reference in archive_members, row.mask_reference

            image_path = local_ps_root / row.image_reference
            mask_path = local_ps_root / row.mask_reference
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            if not image_path.exists() or sha256_path(image_path) != row.sha256:
                with archive.open(row.image_reference) as source, image_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
            assert sha256_path(image_path) == row.sha256, row.record_id

            mask_bytes = archive.read(row.mask_reference)
            source_mask_hash = hashlib.sha256(mask_bytes).hexdigest()
            source_mask_hashes[row.record_id] = source_mask_hash
            if not mask_path.exists() or sha256_path(mask_path) != source_mask_hash:
                mask_path.write_bytes(mask_bytes)
            assert sha256_path(mask_path) == source_mask_hash, row.record_id
            with Image.open(mask_path) as mask:
                mask.verify()

            local_paths[row.record_id] = (str(image_path), str(mask_path))
        else:
            source_path = Path(row.image_reference)
            assert source_path.exists(), source_path
            relative_path = source_path.relative_to(plantdoc_root)
            image_path = local_pd_root / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists() or sha256_path(image_path) != row.sha256:
                shutil.copy2(source_path, image_path)
            assert sha256_path(image_path) == row.sha256, row.record_id
            local_paths[row.record_id] = (str(image_path), None)

        if number % 500 == 0:
            print(f"Verified local cache {number}/{len(all_rows)}")

for role, frame in manifests.items():
    manifests[role] = frame.copy()
    manifests[role]["local_image"] = frame["record_id"].map(
        lambda value: local_paths[value][0]
    )
    manifests[role]["local_mask"] = frame["record_id"].map(
        lambda value: local_paths[value][1]
    )

print(f"Verified {len(all_rows)} unique clean source images from the original datasets.")
print("Source data were read only and were not modified.")

DATA_PROVENANCE = {
    "protocol_fingerprint": PROTOCOL_FINGERPRINT,
    "manifest_sha256": {
        role: sha256_path(manifest_root / filename)
        for role, filename in MANIFEST_FILES.items()
    },
    "class_mapping_sha256": sha256_path(mapping_path),
    "source_mask_hashes_sha256": canonical_hash(source_mask_hashes),
    "class_names": CLASS_NAMES,
}
DATA_CONTENT_FINGERPRINT = canonical_hash(DATA_PROVENANCE)
atomic_json(ANALYSIS_ROOT / 'verified_data_provenance.json', DATA_PROVENANCE)


import re
wrong = manifests["plantseg_test"]["mask_reference"].nunique()
print("Distinct mask_reference values in the locked PlantSeg-test manifest:", wrong)
corrected_rows = []
with zipfile.ZipFile(plantseg_zip) as archive:
    members = set(archive.namelist())
    local_masks = []
    for row in manifests["plantseg_test"].itertuples(index=False):
        reference = re.sub(r"images/test/(.*)\.jpg$", r"annotations/test/\1.png", row.image_reference)
        assert reference != row.image_reference and reference in members, (row.image_reference, reference)
        target = Path(CONFIG["local_cache_root"]) / "plantseg_corrected_test_masks" / Path(reference).name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read(reference)
        target.write_bytes(data)
        local_masks.append(str(target))
        corrected_rows.append({"record_id": row.record_id, "image_reference": row.image_reference,
                               "corrected_mask_reference": reference, "mask_sha256": hashlib.sha256(data).hexdigest()})
corrected = pd.DataFrame(corrected_rows)
assert corrected.corrected_mask_reference.is_unique and len(corrected) == 435
manifests["plantseg_test"] = manifests["plantseg_test"].copy()
manifests["plantseg_test"]["local_mask"] = local_masks
fractions = [float((np.asarray(Image.open(p).convert("L")) > 0).mean()) for p in local_masks]
print("Corrected test-mask lesion fraction quantiles (5/25/50/75/95):", np.percentile(fractions, [5, 25, 50, 75, 95]).round(3))
OUT = Path(CONFIG["drive_runs_root"]); OUT.mkdir(parents=True, exist_ok=True)
corrected.to_csv(OUT / "corrected_plantseg_test_masks.csv", index=False)


def resize_and_pad(image, size, interpolation, fill):
    width, height = image.size
    scale = float(size) / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = TF.resize(
        image,
        [new_height, new_width],
        interpolation=interpolation,
        antialias=(interpolation != InterpolationMode.NEAREST),
    )
    left = (size - new_width) // 2
    right = size - new_width - left
    top = (size - new_height) // 2
    bottom = size - new_height - top
    return TF.pad(image, [left, top, right, bottom], fill=fill)


class PairedStrongTrainTransform:
    def __init__(self, size):
        self.size = size
        self.jitter = ColorJitter(
            brightness=0.30,
            contrast=0.25,
            saturation=0.25,
            hue=0.04,
        )

    def __call__(self, image, mask):
        image = resize_and_pad(
            image, self.size, InterpolationMode.BICUBIC, (124, 116, 104)
        )
        mask = resize_and_pad(
            mask, self.size, InterpolationMode.NEAREST, 0
        )

        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < 0.10:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if random.random() < 0.75:
            angle = random.uniform(-20.0, 20.0)
            translate = [
                int(random.uniform(-0.05, 0.05) * self.size),
                int(random.uniform(-0.05, 0.05) * self.size),
            ]
            scale = random.uniform(0.90, 1.10)
            shear = [random.uniform(-5.0, 5.0), random.uniform(-3.0, 3.0)]
            image = TF.affine(
                image, angle, translate, scale, shear,
                interpolation=InterpolationMode.BICUBIC,
                fill=(124, 116, 104),
            )
            mask = TF.affine(
                mask, angle, translate, scale, shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
        if random.random() < 0.70:
            image = self.jitter(image)
        if random.random() < 0.10:
            image = TF.gaussian_blur(
                image, kernel_size=3, sigma=[0.1, 1.2]
            )

        image = TF.normalize(
            TF.to_tensor(image),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        mask = (TF.pil_to_tensor(mask).float() > 0).float()
        return image, mask


class PairedEvalTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask=None):
        image = resize_and_pad(
            image, self.size, InterpolationMode.BICUBIC, (124, 116, 104)
        )
        image = TF.normalize(
            TF.to_tensor(image),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
        if mask is None:
            return image
        mask = resize_and_pad(
            mask, self.size, InterpolationMode.NEAREST, 0
        )
        mask = (TF.pil_to_tensor(mask).float() > 0).float()
        return image, mask


class ManifestPlantSegDataset(Dataset):
    def __init__(self, frame, size, train):
        self.frame = frame.reset_index(drop=True)
        self.transform = (
            PairedStrongTrainTransform(size)
            if train else PairedEvalTransform(size)
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.local_image) as image_file:
            image = image_file.convert("RGB")
        with Image.open(row.local_mask) as mask_file:
            mask = mask_file.convert("L")
        image, mask = self.transform(image, mask)
        return image, mask, int(row.label), row.record_id


class ManifestPlantDocDataset(Dataset):
    def __init__(self, frame, size):
        self.frame = frame.reset_index(drop=True)
        self.transform = PairedEvalTransform(size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.local_image) as image_file:
            image = self.transform(image_file.convert("RGB"))
        return image, int(row.label), row.record_id

train_labels = manifests["train"]["label"].to_numpy(dtype=int)
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
inverse_frequency = 1.0 / np.maximum(class_counts, 1)
sample_weights = inverse_frequency[train_labels]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def common_loader_kwargs(spec):
    return {
        "batch_size": spec["batch_size"],
        "num_workers": CONFIG["num_workers"],
        "pin_memory": True,
        "persistent_workers": CONFIG["num_workers"] > 0,
    }


def make_train_loader(spec, epoch_seed):
    dataset = ManifestPlantSegDataset(
        manifests["train"], spec["img_size"], train=True
    )
    sampler_generator = torch.Generator().manual_seed(epoch_seed)
    worker_generator = torch.Generator().manual_seed(epoch_seed + 10000)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_generator,
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=worker_generator,
        **common_loader_kwargs(spec),
    )


def make_validation_loader(spec):
    dataset = ManifestPlantSegDataset(
        manifests["validation"], spec["img_size"], train=False
    )
    return DataLoader(
        dataset, shuffle=False, **common_loader_kwargs(spec)
    )


def make_final_evaluation_loaders(spec):
    return {
        "plantseg_official_test": (
            DataLoader(
                ManifestPlantSegDataset(
                    manifests["plantseg_test"], spec["img_size"], train=False
                ),
                shuffle=False,
                **common_loader_kwargs(spec),
            ),
            True,
        ),
        "plantdoc_external_clean": (
            DataLoader(
                ManifestPlantDocDataset(
                    manifests["plantdoc_external"], spec["img_size"]
                ),
                shuffle=False,
                **common_loader_kwargs(spec),
            ),
            False,
        ),
    }


display(pd.DataFrame({
    "class": CLASS_NAMES,
    "optimization_train": class_counts,
}))
print("Validation loaders are used for checkpoint selection.")
print("Official test loaders are not created until the locked final evaluation cell.")

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, reduction="mean"):
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(1)
        dice = (2.0 * intersection + self.smooth) / (
            probabilities.sum(1) + targets.sum(1) + self.smooth
        )
        losses = 1.0 - dice
        return losses if reduction == "none" else losses.mean()

dice_criterion = DiceLoss()

def occupancy_target(masks, size):
    if masks.shape[-2:] == tuple(size):
        return masks
    return F.adaptive_avg_pool2d(masks, output_size=size)

def single_scale_mask_loss(logits, masks):
    target = occupancy_target(masks, logits.shape[-2:])
    flat_target = target.flatten(1)
    positive_mass = flat_target.sum(1).detach()
    pixels = float(flat_target.shape[1])
    valid = positive_mass > 0
    if not bool(valid.any()):
        return logits.sum() * 0.0

    negative_mass = pixels - positive_mass
    positive_weight = (
        negative_mass / positive_mass.clamp_min(1e-6)
    ).clamp(
        1.0, CONFIG["mask_pos_weight_cap"]
    ).view(-1, 1, 1, 1)

    bce_map = -(
        positive_weight * target * F.logsigmoid(logits)
        + (1.0 - target) * F.logsigmoid(-logits)
    )
    bce = bce_map.flatten(1).mean(1)[valid].mean()
    dice = dice_criterion(logits, target, reduction="none")[valid].mean()
    return (
        CONFIG["mask_bce_weight"] * bce
        + CONFIG["mask_dice_weight"] * dice
    )

def combined_mask_loss(logits_by_scale, masks):
    if not logits_by_scale:
        return masks.sum() * 0.0
    losses = [
        single_scale_mask_loss(logits, masks)
        for logits in logits_by_scale.values()
    ]
    return torch.stack(losses).mean()

def scale_consistency_loss(logits_by_scale):
    if not logits_by_scale or len(logits_by_scale) < 2:
        first = next(iter(logits_by_scale.values()), None) if logits_by_scale else None
        return first.sum() * 0.0 if first is not None else None
    target_size = logits_by_scale["fine"].shape[-2:]
    probabilities = []
    for logits in logits_by_scale.values():
        probability = torch.sigmoid(logits)
        if probability.shape[-2:] != target_size:
            probability = F.interpolate(
                probability,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        probabilities.append(probability)
    consensus = torch.stack(probabilities, dim=0).mean(dim=0)
    return torch.stack([
        F.mse_loss(probability, consensus)
        for probability in probabilities
    ]).mean()

@torch.no_grad()
def evaluate(model, loader, dataset_name, has_masks=False, save_predictions=False):
    model.eval()
    predictions, targets, paths = [], [], []
    intersection = union = predicted_pixels = target_pixels = 0.0
    gate_values = []

    for batch in loader:
        if has_masks:
            images, masks, labels, batch_paths = batch
            masks = masks.to(DEVICE, non_blocking=True)
        else:
            images, labels, batch_paths = batch
            masks = None
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        outputs = model(images)
        predicted_labels = outputs["logits"].argmax(1)

        predictions.extend(predicted_labels.cpu().tolist())
        targets.extend(labels.cpu().tolist())
        paths.extend(batch_paths)

        if outputs["fusion_gates"] is not None:
            gate_values.append(outputs["fusion_gates"].cpu().numpy())

        if has_masks and outputs["mask_logits"] is not None:
            predicted_mask = torch.sigmoid(outputs["mask_logits"])
            predicted_mask = F.interpolate(
                predicted_mask,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ) > 0.5
            target_mask = masks > 0.5
            intersection += (predicted_mask & target_mask).sum().item()
            union += (predicted_mask | target_mask).sum().item()
            predicted_pixels += predicted_mask.sum().item()
            target_pixels += target_mask.sum().item()

    predictions_array = np.asarray(predictions)
    targets_array = np.asarray(targets)
    result = {
        "dataset": dataset_name,
        "accuracy": float(100.0 * accuracy_score(targets_array, predictions_array)),
        "balanced_accuracy": float(100.0 * balanced_accuracy_score(targets_array, predictions_array)),
        "macro_f1": float(100.0 * f1_score(targets_array, predictions_array, average="macro", zero_division=0)),
        "macro_precision": float(100.0 * precision_score(targets_array, predictions_array, average="macro", zero_division=0)),
        "macro_recall": float(100.0 * recall_score(targets_array, predictions_array, average="macro", zero_division=0)),
    }
    if target_pixels > 0:
        result["mask_iou"] = float(100.0 * intersection / (union + 1e-8))
        result["mask_dice"] = float(
            200.0 * intersection / (predicted_pixels + target_pixels + 1e-8)
        )
    if gate_values:
        gates = np.concatenate(gate_values, axis=0)
        result["gate_fine"] = float(gates[:, 0].mean())
        result["gate_mid"] = float(gates[:, 1].mean())
        result["gate_coarse"] = float(gates[:, 2].mean())
    if save_predictions:
        result["predictions"] = predictions
        result["targets"] = targets
        result["paths"] = paths
    return result

def print_metrics(result):
    message = (
        f"{result['dataset']}: Acc={result['accuracy']:.2f} "
        f"BalAcc={result['balanced_accuracy']:.2f} "
        f"F1={result['macro_f1']:.2f}"
    )
    if "mask_dice" in result:
        message += (
            f" MaskDice={result['mask_dice']:.2f}"
            f" MaskIoU={result['mask_iou']:.2f}"
        )
    if "gate_fine" in result:
        message += (
            f" Gates=[{result['gate_fine']:.3f},"
            f"{result['gate_mid']:.3f},{result['gate_coarse']:.3f}]"
        )
    print(message)


MODEL_DIR = CONFIG["model_dir"]
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    snapshot_download(repo_id=CONFIG["model_repo"], local_dir=MODEL_DIR, revision="3db342a1b00a91c350dca4df93a1645ad2ac9493")

def load_pretrained_vmamba():
    return AutoModelForImageClassification.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )


def empty_outputs(logits):
    return {
        "logits": logits,
        "mask_logits": None,
        "mask_logits_by_scale": None,
        "soft_mask": None,
        "fusion_gates": None,
    }


class VMambaBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pretrained = load_pretrained_vmamba()
        self.vmamba = pretrained.vmamba
        self.feature_dim = self.vmamba.num_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.uses_consistency = False

    def parameter_groups(self):
        return list(self.vmamba.parameters()), list(self.classifier.parameters())

    def forward(self, images):
        x = self.vmamba.patch_embeddings(images)
        for layer in self.vmamba.layers:
            x = layer(x)
        x = self.vmamba.norm(x)
        vector = self.vmamba.avgpool(x).flatten(1)
        return empty_outputs(self.classifier(vector))


class NativeMeanFusionHead(nn.Module):
    SCALE_NAMES = ("fine", "mid", "coarse")

    def __init__(self, channels=(192, 384, 768), hidden=128):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_in, hidden, 1, bias=False),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
            for channels_in in channels
        ])
        self.mask_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            )
            for _ in channels
        ])

    @staticmethod
    def weighted_pool(feature, weight):
        numerator = (feature * weight).sum(dim=(2, 3))
        denominator = weight.sum(dim=(2, 3)).clamp_min(1e-6)
        return numerator / denominator

    def forward(self, features):
        logits_by_scale = {}
        lesion_vectors = []
        for name, projection, predictor, feature in zip(
            self.SCALE_NAMES,
            self.projections,
            self.mask_predictors,
            features,
        ):
            projected = projection(feature)
            logits = predictor(projected)
            probability = torch.sigmoid(logits)
            logits_by_scale[name] = logits
            lesion_vectors.append(
                self.weighted_pool(projected, probability)
            )
        return logits_by_scale, lesion_vectors


def mean_probability_map(logits_by_scale):
    target_size = logits_by_scale["fine"].shape[-2:]
    probabilities = []
    for name in ("fine", "mid", "coarse"):
        probability = torch.sigmoid(logits_by_scale[name])
        if probability.shape[-2:] != target_size:
            probability = F.interpolate(
                probability,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        probabilities.append(probability)
    return torch.stack(probabilities, dim=0).mean(dim=0)


class CurrentMeanFusionVMamba(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pretrained = load_pretrained_vmamba()
        self.vmamba = pretrained.vmamba
        self.feature_dim = self.vmamba.num_features
        self.region_head = NativeMeanFusionHead()
        self.lesion_projection = nn.Sequential(
            nn.Linear(128 * 3, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.fusion_norm = nn.LayerNorm(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.uses_consistency = False

    def parameter_groups(self):
        encoder = list(self.vmamba.parameters())
        encoder_ids = {id(parameter) for parameter in encoder}
        head = [
            parameter for parameter in self.parameters()
            if id(parameter) not in encoder_ids
        ]
        return encoder, head

    def forward(self, images):
        x = self.vmamba.patch_embeddings(images)
        features = []
        for layer_index in range(3):
            x = self.vmamba.layers[layer_index](x)
            features.append(x)

        logits_by_scale, lesion_vectors = self.region_head(features)
        soft_mask = mean_probability_map(logits_by_scale)
        mask_logits = torch.logit(
            soft_mask.clamp(1e-6, 1.0 - 1e-6)
        )

        x = self.vmamba.layers[3](x)
        final_map = self.vmamba.norm(x)
        global_vector = self.vmamba.avgpool(final_map).flatten(1)
        lesion_vector = self.lesion_projection(
            torch.cat(lesion_vectors, dim=1)
        )
        classification_vector = self.fusion_norm(
            global_vector + lesion_vector
        )
        return {
            "logits": self.classifier(classification_vector),
            "mask_logits": mask_logits,
            "mask_logits_by_scale": logits_by_scale,
            "soft_mask": soft_mask,
            "fusion_gates": None,
        }


def build_model(name):
    if name == "vmamba_baseline":
        return VMambaBaseline(NUM_CLASSES)
    if name == "current_full":
        return CurrentMeanFusionVMamba(NUM_CLASSES)
    raise ValueError(f"Unknown model: {name}")

class PlainMultitaskVMamba(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pretrained = load_pretrained_vmamba()
        self.vmamba = pretrained.vmamba
        self.feature_dim = self.vmamba.num_features
        self.region_head = NativeMeanFusionHead()
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def parameter_groups(self):
        encoder = list(self.vmamba.parameters())
        encoder_ids = {id(parameter) for parameter in encoder}
        head = [
            parameter for parameter in self.parameters()
            if id(parameter) not in encoder_ids
        ]
        return encoder, head

    def forward(self, images):
        x = self.vmamba.patch_embeddings(images)
        features = []
        for layer_index in range(3):
            x = self.vmamba.layers[layer_index](x)
            features.append(x)

        logits_by_scale, _ = self.region_head(features)
        soft_mask = mean_probability_map(logits_by_scale)
        mask_logits = torch.logit(soft_mask.clamp(1e-6, 1.0 - 1e-6))

        x = self.vmamba.layers[3](x)
        final_map = self.vmamba.norm(x)
        global_vector = self.vmamba.avgpool(final_map).flatten(1)

        return {
            "logits": self.classifier(global_vector),
            "mask_logits": mask_logits,
            "mask_logits_by_scale": logits_by_scale,
            "soft_mask": soft_mask,
            "fusion_gates": None,
        }


class NativeSingleScaleHead(nn.Module):
    def __init__(self, channels_in, hidden=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels_in, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    @staticmethod
    def weighted_pool(feature, weight):
        numerator = (feature * weight).sum(dim=(2, 3))
        denominator = weight.sum(dim=(2, 3)).clamp_min(1e-6)
        return numerator / denominator

    def forward(self, feature):
        projected = self.projection(feature)
        logits = self.mask_predictor(projected)
        probability = torch.sigmoid(logits)
        lesion_vector = self.weighted_pool(projected, probability)
        return logits, lesion_vector


class SingleScaleFusionVMamba(nn.Module):
    def __init__(self, num_classes, scale_name, scale_index, scale_channels):
        super().__init__()
        self.scale_name = scale_name
        self.scale_index = int(scale_index)

        pretrained = load_pretrained_vmamba()
        self.vmamba = pretrained.vmamba
        self.feature_dim = self.vmamba.num_features

        self.region_head = NativeSingleScaleHead(scale_channels)
        self.lesion_projection = nn.Sequential(
            nn.Linear(128, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.fusion_norm = nn.LayerNorm(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def parameter_groups(self):
        encoder = list(self.vmamba.parameters())
        encoder_ids = {id(parameter) for parameter in encoder}
        head = [
            parameter for parameter in self.parameters()
            if id(parameter) not in encoder_ids
        ]
        return encoder, head

    def forward(self, images):
        x = self.vmamba.patch_embeddings(images)
        features = []
        for layer_index in range(3):
            x = self.vmamba.layers[layer_index](x)
            features.append(x)

        mask_logits, lesion_native = self.region_head(
            features[self.scale_index]
        )
        logits_by_scale = {self.scale_name: mask_logits}

        x = self.vmamba.layers[3](x)
        final_map = self.vmamba.norm(x)
        global_vector = self.vmamba.avgpool(final_map).flatten(1)
        lesion_vector = self.lesion_projection(lesion_native)
        classification_vector = self.fusion_norm(global_vector + lesion_vector)

        return {
            "logits": self.classifier(classification_vector),
            "mask_logits": mask_logits,
            "mask_logits_by_scale": logits_by_scale,
            "soft_mask": torch.sigmoid(mask_logits),
            "fusion_gates": None,
        }


SINGLE = {"fine": ("fine", 0, 192), "mid": ("mid", 1, 384), "coarse": ("coarse", 2, 768)}
EVAL_ROOT = STORAGE + "/msls_vmamba/runs/seed11_analyses"
SEED11_MODELS = {
    "vmamba_baseline": (lambda: VMambaBaseline(NUM_CLASSES), STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1/cleanv1_vmamba_baseline_strong_384_seed11/best_model.pth"),
    "current_full": (lambda: CurrentMeanFusionVMamba(NUM_CLASSES), STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1/cleanv1_current_full_strong_384_seed11/best_model.pth"),
    "current_full_no_mask_loss": (lambda: CurrentMeanFusionVMamba(NUM_CLASSES),
                                  f"{EVAL_ROOT}/mask_controls_seed11_v1/cleanv1_current_full_no_mask_loss_384_seed11/best_model.pth"),
    "current_full_same_class_permuted_masks": (lambda: CurrentMeanFusionVMamba(NUM_CLASSES),
                                  f"{EVAL_ROOT}/mask_controls_seed11_v1/cleanv1_current_full_same_class_permuted_masks_384_seed11/best_model.pth"),
    "vmamba_plain_multitask": (lambda: PlainMultitaskVMamba(NUM_CLASSES),
                                  f"{EVAL_ROOT}/plain_multitask_control_seed11_v1/cleanv1_vmamba_plain_multitask_384_seed11/best_model.pth"),
}
for scale, settings in SINGLE.items():
    SEED11_MODELS[f"current_full_{scale}_only"] = (
        (lambda s=settings: SingleScaleFusionVMamba(NUM_CLASSES, *s)),
        f"{EVAL_ROOT}/single_scale_ablation_seed11_v1/cleanv1_current_full_{scale}_only_384_seed11/best_model.pth")
missing = [name for name, (_, path) in SEED11_MODELS.items() if not Path(path).exists()]
assert not missing, f"Seed-11 checkpoints missing (finish the seed-11 ablation trainings first): {missing}"
print("All", len(SEED11_MODELS), "seed-11 checkpoints found.")


@torch.no_grad()
def per_image_overlap(model, loader):
    model.eval(); dice, iou = [], []
    for images, masks, labels, ids in loader:
        outputs = model(images.to(DEVICE))
        if outputs["mask_logits"] is None:
            return None
        predicted = F.interpolate(torch.sigmoid(outputs["mask_logits"]), size=masks.shape[-2:], mode="bilinear",
                                  align_corners=False) > 0.5
        target = masks.to(DEVICE) > 0.5
        inter = (predicted & target).flatten(1).sum(1).float(); union = (predicted | target).flatten(1).sum(1).float()
        total = predicted.flatten(1).sum(1).float() + target.flatten(1).sum(1).float()
        dice += (200.0 * inter / total.clamp_min(1)).tolist(); iou += (100.0 * inter / union.clamp_min(1)).tolist()
    return {"mean_per_image_dice": float(np.mean(dice)), "mean_per_image_iou": float(np.mean(iou)), "n_images": len(dice)}


spec = {"img_size": 384, "batch_size": 16}
results = {}
access = []
for name, (builder, path) in SEED11_MODELS.items():
    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["protocol_fingerprint"] == PROTOCOL_FINGERPRINT
    model = builder().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    entry = {"checkpoint": path, "checkpoint_sha256": sha256_path(path), "best_epoch": int(checkpoint["epoch"]),
             "parameters": int(sum(p.numel() for p in model.parameters()))}
    entry["optimization_validation"] = evaluate(model, make_validation_loader(spec), "validation", has_masks=True, save_predictions=True)
    entry["validation_per_image_overlap"] = per_image_overlap(model, make_validation_loader(spec))
    for split, (loader, has_masks) in make_final_evaluation_loaders(spec).items():
        entry[split] = evaluate(model, loader, split, has_masks=has_masks, save_predictions=True)
        access.append({"utc": utc_now(), "model": name, "split": split, "n_images": len(entry[split]["targets"])})
        if split == "plantseg_official_test":
            entry["plantseg_test_per_image_overlap_corrected_masks"] = per_image_overlap(model, loader)
    results[name] = entry
    atomic_json(OUT / f"{name}_seed11_locked_eval.json", entry)
    print(name, {k: round(v["macro_f1"], 2) for k, v in entry.items() if isinstance(v, dict) and "macro_f1" in v})
    del model; torch.cuda.empty_cache()
atomic_json(OUT / "test_access_log.json", access)

consistency = []
for name, folder in (("vmamba_baseline", "cleanv1_vmamba_baseline_strong_384_seed11"), ("current_full", "cleanv1_current_full_strong_384_seed11")):
    stored = load_json(Path(STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1") / folder / "final_locked_summary.json")
    for split in ("optimization_validation", "plantseg_official_test", "plantdoc_external_clean"):
        new, old = results[name][split], stored[split]
        assert new["paths"] == old["paths"]
        agree = float(np.mean(np.asarray(new["predictions"]) == np.asarray(old["predictions"])))
        consistency.append({"model": name, "split": split, "agreement": agree,
                            "macro_f1_stored": old["macro_f1"], "macro_f1_recomputed": new["macro_f1"],
                            "mismatched_ids": [p for p, a, b in zip(new["paths"], new["predictions"], old["predictions"]) if a != b]})
        assert agree >= 0.995, (name, split, agree)
atomic_json(OUT / "consistency_with_main_training.json", {"gpu": torch.cuda.get_device_name(0), "checks": consistency})
print("Consistency with the main training runs:", [(c["model"], c["split"], round(c["agreement"], 4)) for c in consistency])

rows = []
for name, entry in results.items():
    row = {"model": name, "parameters": entry["parameters"], "best_epoch": entry["best_epoch"]}
    for split in ("optimization_validation", "plantseg_official_test", "plantdoc_external_clean"):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "mask_dice", "mask_iou"):
            if metric in entry[split]:
                row[f"{split}_{metric}"] = entry[split][metric]
    for key in ("validation_per_image_overlap", "plantseg_test_per_image_overlap_corrected_masks"):
        if entry.get(key):
            row[f"{key}_dice"] = entry[key]["mean_per_image_dice"]; row[f"{key}_iou"] = entry[key]["mean_per_image_iou"]
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(OUT / "seed11_locked_eval_summary.csv", index=False)
display(table.round(2))


RUN_SPECS = [{"name": "current_full_strong_384", "model": "current_full", "img_size": 384, "batch_size": 16,
              "accumulation_steps": 2, "label_smoothing": 0.10, "warmup_epochs": 4, "encoder_lr": 5e-5, "head_lr": 5e-4}]
CONFIG["seed"] = 11

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


diagnostic_spec = next(
    spec for spec in RUN_SPECS
    if spec["name"] == "current_full_strong_384"
)

run_name = (
    "cleanv1_current_full_strong_384_"
    f"seed{CONFIG['seed']}"
)

run_directory = (
    Path(STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1")
    / run_name
)

checkpoint_path = (
    run_directory / "best_model.pth"
)

assert checkpoint_path.exists(), (
    f"Missing checkpoint: {checkpoint_path}"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
)

assert (
    checkpoint["protocol_fingerprint"]
    == PROTOCOL_FINGERPRINT
), "Checkpoint and clean manifests do not match."

model = build_model(
    "current_full"
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

validation_loader = make_validation_loader(
    diagnostic_spec
)


global_vectors = []
lesion_vectors = []
all_targets = []
all_record_ids = []

with torch.no_grad():
    for images, masks, labels, record_ids in (
        validation_loader
    ):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        x = model.vmamba.patch_embeddings(
            images
        )

        multiscale_features = []

        for layer_index in range(3):
            x = model.vmamba.layers[
                layer_index
            ](x)

            multiscale_features.append(x)

        (
            mask_logits_by_scale,
            native_lesion_vectors,
        ) = model.region_head(
            multiscale_features
        )

        lesion_vector = model.lesion_projection(
            torch.cat(
                native_lesion_vectors,
                dim=1,
            )
        )

        x = model.vmamba.layers[3](x)

        final_feature_map = model.vmamba.norm(
            x
        )

        global_vector = model.vmamba.avgpool(
            final_feature_map
        ).flatten(1)

        global_vectors.append(
            global_vector.cpu()
        )

        lesion_vectors.append(
            lesion_vector.cpu()
        )

        all_targets.extend(
            labels.tolist()
        )

        all_record_ids.extend(
            list(record_ids)
        )


global_vectors = torch.cat(
    global_vectors,
    dim=0,
).to(DEVICE)

lesion_vectors = torch.cat(
    lesion_vectors,
    dim=0,
).to(DEVICE)

targets = np.asarray(
    all_targets,
    dtype=int,
)

target_tensor = torch.tensor(
    targets,
    device=DEVICE,
    dtype=torch.long,
)


class_counts = np.bincount(
    targets,
    minlength=NUM_CLASSES,
)

largest_class = int(
    class_counts.max()
)

sorted_indices = np.argsort(
    targets,
    kind="stable",
)

rolled_indices = np.roll(
    sorted_indices,
    largest_class,
)

lesion_permutation = np.empty(
    len(targets),
    dtype=int,
)

lesion_permutation[
    sorted_indices
] = rolled_indices

assert np.all(
    targets[lesion_permutation]
    != targets
), "The shuffled lesion vector must come from a different class."

lesion_permutation_tensor = torch.tensor(
    lesion_permutation,
    device=DEVICE,
    dtype=torch.long,
)

shuffled_lesion_vectors = lesion_vectors[
    lesion_permutation_tensor
]


with torch.no_grad():
    condition_logits = {
        "global_plus_correct_lesion":
            model.classifier(
                model.fusion_norm(
                    global_vectors
                    + lesion_vectors
                )
            ),

        "global_only":
            model.classifier(
                model.fusion_norm(
                    global_vectors
                )
            ),

        "global_plus_shuffled_lesion":
            model.classifier(
                model.fusion_norm(
                    global_vectors
                    + shuffled_lesion_vectors
                )
            ),

        "lesion_only":
            model.classifier(
                model.fusion_norm(
                    lesion_vectors
                )
            ),
    }


crop_names = sorted({
    class_name.split("_", 1)[0]
    for class_name in CLASS_NAMES
})

crop_to_index = {
    crop_name: index
    for index, crop_name in enumerate(
        crop_names
    )
}

class_to_crop = np.asarray([
    crop_to_index[
        class_name.split("_", 1)[0]
    ]
    for class_name in CLASS_NAMES
])

target_crops = class_to_crop[
    targets
]

condition_predictions = {}
condition_true_probabilities = {}

for condition_name, logits in (
    condition_logits.items()
):
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    condition_predictions[
        condition_name
    ] = (
        logits.argmax(dim=1)
        .cpu()
        .numpy()
    )

    condition_true_probabilities[
        condition_name
    ] = (
        probabilities.gather(
            1,
            target_tensor.unsqueeze(1),
        )
        .squeeze(1)
        .cpu()
        .numpy()
    )


reference_name = (
    "global_plus_correct_lesion"
)

reference_predictions = (
    condition_predictions[
        reference_name
    ]
)

reference_macro_f1 = (
    100.0
    * f1_score(
        targets,
        reference_predictions,
        average="macro",
        zero_division=0,
    )
)

reference_true_probability = float(
    condition_true_probabilities[
        reference_name
    ].mean()
)

result_rows = []

for condition_name in condition_logits:
    predictions_array = (
        condition_predictions[
            condition_name
        ]
    )

    true_probabilities = (
        condition_true_probabilities[
            condition_name
        ]
    )

    macro_f1 = (
        100.0
        * f1_score(
            targets,
            predictions_array,
            average="macro",
            zero_division=0,
        )
    )

    result_rows.append({
        "condition":
            condition_name,

        "accuracy":
            100.0
            * accuracy_score(
                targets,
                predictions_array,
            ),

        "balanced_accuracy":
            100.0
            * balanced_accuracy_score(
                targets,
                predictions_array,
            ),

        "macroF1":
            macro_f1,

        "crop_accuracy":
            100.0
            * accuracy_score(
                target_crops,
                class_to_crop[
                    predictions_array
                ],
            ),

        "mean_true_class_probability":
            100.0
            * true_probabilities.mean(),

        "agreement_with_full":
            100.0
            * np.mean(
                predictions_array
                == reference_predictions
            ),

        "delta_macroF1_from_full":
            macro_f1
            - reference_macro_f1,

        "true_probability_drop_from_full":
            100.0
            * (
                reference_true_probability
                - true_probabilities.mean()
            ),
    })


branch_dependency_table = pd.DataFrame(
    result_rows
)

display(
    branch_dependency_table.round(3)
)


global_norm = (
    global_vectors.norm(dim=1)
    .mean()
    .item()
)

lesion_norm = (
    lesion_vectors.norm(dim=1)
    .mean()
    .item()
)

with torch.no_grad():
    full_true_logits = (
        condition_logits[
            "global_plus_correct_lesion"
        ]
        .gather(
            1,
            target_tensor.unsqueeze(1),
        )
        .squeeze(1)
    )

    global_only_true_logits = (
        condition_logits[
            "global_only"
        ]
        .gather(
            1,
            target_tensor.unsqueeze(1),
        )
        .squeeze(1)
    )

    shuffled_true_logits = (
        condition_logits[
            "global_plus_shuffled_lesion"
        ]
        .gather(
            1,
            target_tensor.unsqueeze(1),
        )
        .squeeze(1)
    )


print(
    "\nMean global-vector norm:",
    f"{global_norm:.3f}"
)

print(
    "Mean lesion-vector norm:",
    f"{lesion_norm:.3f}"
)

print(
    "Lesion/global norm ratio:",
    f"{lesion_norm / max(global_norm, 1e-8):.3f}"
)

print(
    "Mean correct-class logit gain "
    "from correct lesion vector:",
    f"{(full_true_logits - global_only_true_logits).mean().item():.3f}"
)

print(
    "Mean correct-class logit advantage "
    "over shuffled lesion vector:",
    f"{(full_true_logits - shuffled_true_logits).mean().item():.3f}"
)


analysis_directory = (
    Path(STORAGE + "/msls_vmamba/runs/seed11_analyses")
    / "internal_branch_diagnostic_seed11"
)

analysis_directory.mkdir(
    parents=True,
    exist_ok=True,
)

branch_dependency_table.to_csv(
    analysis_directory
    / "internal_lesion_branch_summary.csv",
    index=False,
)

per_image_results = pd.DataFrame({
    "record_id":
        all_record_ids,

    "true_class": [
        CLASS_NAMES[label]
        for label in targets
    ],

    "shuffled_lesion_source_class": [
        CLASS_NAMES[label]
        for label in targets[
            lesion_permutation
        ]
    ],
})

for condition_name in condition_logits:
    per_image_results[
        f"{condition_name}_prediction"
    ] = [
        CLASS_NAMES[label]
        for label in condition_predictions[
            condition_name
        ]
    ]

    per_image_results[
        f"{condition_name}_true_probability"
    ] = condition_true_probabilities[
        condition_name
    ]


per_image_results.to_csv(
    analysis_directory
    / "internal_lesion_branch_per_image.csv",
    index=False,
)

print(
    "\nSaved:",
    analysis_directory
)

del model
del global_vectors
del lesion_vectors
del shuffled_lesion_vectors
del condition_logits

torch.cuda.empty_cache()
