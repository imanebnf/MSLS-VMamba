# Trains MSVMamba-Tiny with five seeds; fills runs/msvmamba_tiny_seeds11-55.
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
    "drive_runs_root": STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/msvmamba_tiny_11_55_v1",
    "reference_runs_root": STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1",
    "exif_policy": "original_uncorrected",
    "comparator_repo": "https://github.com/YuHengsss/MSVMamba.git",
    "comparator_commit": "c39401918e94f11eb29e4932b901b3fd139f1377",
    "comparator_config": "classification/configs/ms_vssm/vssm_tiny_224_ms.yaml",
    "comparator_checkpoint_sha256": "6b4e4f3fcbfc98a3098bf5fedbd17f73aff61167fefe56339f17bf8d1429eb43",
    "comparator_weights": "model_ema",
    "local_cache_root": SCRATCH + "/vmamba_clean_protocol_v1",
    "model_repo": "saurabhati/VMamba_ImageNet_82.6",
    "model_dir": SCRATCH + "/vmamba_msls_model",
    "existing_model_dir": SCRATCH + "/vmamba_model",
    "model_revision": None,
}

RUN_SPECS = [
    {
        "name": "msvmamba_tiny_384",
        "model": "msvmamba_tiny",
        "img_size": 384,
        "batch_size": 16,
        "accumulation_steps": 2,
        "label_smoothing": 0.10,
        "warmup_epochs": 4,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
        "lambda_mask": 0.0,
    },
]
REFERENCE_SPECS = {
    "current_full": {"name": "current_full_strong_384", "model": "current_full",
                     "img_size": 384,
                     "batch_size": 16,
                     "accumulation_steps": 2,
                     "label_smoothing": 0.10,
                     "warmup_epochs": 4,
                     "encoder_lr": 5e-5,
                     "head_lr": 5e-4,},
    "vmamba_baseline": {"name": "vmamba_baseline_strong_384", "model": "vmamba_baseline",
                        "img_size": 384,
                        "batch_size": 16,
                        "accumulation_steps": 2,
                        "label_smoothing": 0.10,
                        "warmup_epochs": 4,
                        "encoder_lr": 5e-5,
                        "head_lr": 5e-4,},
}

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
EXTENSION_VERSION = "msvmamba_tiny_11_55_v1"
EXPECTED_PARAMETERS = {"vmamba_baseline": 29492368, "current_full": 30405907,
                       "current_full_uniform_map": 30405907,
                       "msvmamba_tiny": 32289616}
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
print("Up to 5 new training runs (MSVMamba-Tiny, seeds 11-55). Main training runs are read only.")
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


import re
from PIL import ImageOps
REFERENCE_ROOT = Path(CONFIG["reference_runs_root"])
assert REFERENCE_ROOT.exists(), REFERENCE_ROOT
R1_SOURCE = {
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

# The manifests must match the recorded cleaning protocol before anything is trained.
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


E34_DATA_CONTENT_FINGERPRINT = "5c340193e4b10f1a04cec57254013d448eca32e3622729d063ce8b464aad4a05"
assert DATA_CONTENT_FINGERPRINT == E34_DATA_CONTENT_FINGERPRINT, (
    "Manifests, class mapping or masks differ from the main training runs: stop and inspect.")
print("Data content fingerprint equals the main training runs:", DATA_CONTENT_FINGERPRINT[:16])


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


def snapshot_file_hashes(directory):
    directory = Path(directory)
    allowed = {'.json', '.py', '.bin', '.safetensors', '.model', '.txt'}
    paths = [path for path in directory.rglob('*') if path.is_file()
             and path.suffix in allowed
             and not any(part.startswith('.') or part == '__pycache__'
                         for part in path.relative_to(directory).parts)]
    assert any(path.name == 'config.json' for path in paths), directory
    assert any(path.suffix in {'.bin', '.safetensors'} for path in paths), (
        'No pretrained weights found', str(directory)
    )
    return {str(path.relative_to(directory)): sha256_path(path)
            for path in sorted(paths)}

def freeze_pretrained_snapshot():
    local = Path(CONFIG['model_dir'])
    frozen = REFERENCE_ROOT / 'pretrained_snapshot'
    manifest_path = REFERENCE_ROOT / 'pretrained_snapshot_manifest.json'
    assert manifest_path.exists(), 'Main training pretrained snapshot manifest not found'
    if manifest_path.exists():
        metadata = load_json(manifest_path)
        assert metadata['model_repo'] == CONFIG['model_repo']
        if CONFIG['model_revision'] is not None:
            assert CONFIG['model_revision'] == metadata.get('revision')
        for relative, expected in metadata['files'].items():
            assert sha256_path(frozen / relative) == expected, relative
        origin = frozen
    else:
        existing = Path(CONFIG['existing_model_dir'])
        if (existing / 'config.json').exists():
            origin = existing
            revision = None
            source_kind = 'existing_local_snapshot_revision_unrecorded'
        else:
            revision = HfApi().model_info(
                CONFIG['model_repo'], revision=CONFIG['model_revision']
            ).sha
            snapshot_download(repo_id=CONFIG['model_repo'], revision=revision,
                              local_dir=str(local))
            origin = local
            source_kind = 'resolved_huggingface_revision'
        files = snapshot_file_hashes(origin)
        metadata = {'model_repo': CONFIG['model_repo'], 'revision': revision,
                    'source_kind': source_kind, 'created_utc': utc_now(),
                    'files': files, 'content_sha256': canonical_hash(files)}
        for relative, expected in files.items():
            target = frozen / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256_path(target) != expected:
                shutil.copy2(origin / relative, target)
            assert sha256_path(target) == expected
        atomic_json(manifest_path, metadata)
        origin = frozen
    for relative, expected in metadata['files'].items():
        target = local / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256_path(target) != expected:
            shutil.copy2(origin / relative, target)
        assert sha256_path(target) == expected, relative
    actual = snapshot_file_hashes(local)
    assert actual == metadata['files'], (
        'The local pretrained directory contains unrecorded or changed model files. '
        'Use a clean CONFIG[model_dir] and run again.'
    )
    return metadata

MODEL_SNAPSHOT = freeze_pretrained_snapshot()
MODEL_SNAPSHOT_SHA256 = MODEL_SNAPSHOT['content_sha256']
print('Frozen pretrained snapshot:', MODEL_SNAPSHOT_SHA256)
print('Recorded revision:', MODEL_SNAPSHOT.get('revision'))


assert MODEL_SNAPSHOT_SHA256 == "dcb966adb18f34f1d050b8eb2d154b488b91a5a7fb787770bbac0535fc449479", "Pretrained snapshot differs from the main training runs"
atomic_json(EXTENSION_ROOT / 'pretrained_snapshot_manifest_reference.json',
            {'reference': str(REFERENCE_ROOT / 'pretrained_snapshot_manifest.json'), 'metadata': MODEL_SNAPSHOT})

MODEL_DIR = CONFIG["model_dir"]

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


import importlib.util
import pickle
import subprocess
import types
import collections
import shutil

MSV_REPO = "https://github.com/YuHengsss/MSVMamba.git"
MSV_COMMIT = "c39401918e94f11eb29e4932b901b3fd139f1377"
MSV_DIR = Path(SCRATCH + "/MSVMamba")
MSV_FILE_SHA256 = {'classification/models/__init__.py': 'b00ab65223c09dfe80216ed4eac40e63629098b9ea3b7f7147a18aa33f305299', 'classification/models/vmamba.py': '09da40b7c38dc9ecb4334ce1654b2826051fba8abfeec140b24ec5bf5c59b011', 'classification/models/utils.py': '67c5f2041c25d4b854c92a8547b2c35e9a7401bf815e5958d87c239b9183919c', 'classification/config.py': 'c852ae8f260a4bee5c2b00d310781ca19d1488fd3407c0b3f22e8b7d6a1de729', 'classification/configs/ms_vssm/vssm_tiny_224_ms.yaml': '182d0a6779e425c58321cc051e9e43234cab39b551404a4d55522f516a2ed983'}
MSV_CHECKPOINT_DRIVE_ID = "1QMrGkb7eZ-_m3DAgwVQBi2GS1FmvfvU5"
MSV_CHECKPOINT_SHA256 = "6b4e4f3fcbfc98a3098bf5fedbd17f73aff61167fefe56339f17bf8d1429eb43"
MSV_CHECKPOINT = EXTENSION_ROOT / "pretrained_msvmamba" / "vssm_tiny_ms_e300.pth"
assert CONFIG["comparator_commit"] == MSV_COMMIT and CONFIG["comparator_checkpoint_sha256"] == MSV_CHECKPOINT_SHA256

if not MSV_DIR.exists():
    subprocess.check_call(["git", "clone", "--quiet", MSV_REPO, str(MSV_DIR)])
subprocess.check_call(["git", "-C", str(MSV_DIR), "checkout", "--quiet", MSV_COMMIT])
assert subprocess.check_output(["git", "-C", str(MSV_DIR), "rev-parse", "HEAD"], text=True).strip() == MSV_COMMIT
for relative, expected in MSV_FILE_SHA256.items():
    assert sha256_path(MSV_DIR / relative) == expected, f"MSVMamba file differs from the reviewed one: {relative}"

if not MSV_CHECKPOINT.exists():
    MSV_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    local_copy = Path(SCRATCH + "/vssm_tiny_ms_e300.pth")
    gdown.download(id=MSV_CHECKPOINT_DRIVE_ID, output=str(local_copy), quiet=False)
    assert local_copy.exists(), (
        "Download failed. Open https://drive.google.com/drive/folders/1ddNWk89ohcxxMuVEaJ2Y5A6xeBKgA7lU, download "
        f"vssm_tiny_ms_e300.pth and upload it to {MSV_CHECKPOINT}, then run again.")
    assert sha256_path(local_copy) == MSV_CHECKPOINT_SHA256, "Downloaded checkpoint differs from the one checked on the server"
    shutil.copy2(local_copy, MSV_CHECKPOINT)
assert sha256_path(MSV_CHECKPOINT) == MSV_CHECKPOINT_SHA256, "Checkpoint differs from the expected SHA-256"


class _StoredConfig(dict):
    def __setstate__(self, state):
        self.update(state if isinstance(state, dict) else {})


_ALLOWED_CLASSES = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("yacs.config", "CfgNode"): _StoredConfig,
    ("__builtin__", "set"): set,
    ("builtins", "set"): set,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_CLASSES:
            return _ALLOWED_CLASSES[(module, name)]
        if module == "torch" and name.endswith("Storage"):
            return getattr(torch, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


_restricted_pickle = types.SimpleNamespace(Unpickler=_RestrictedUnpickler, load=pickle.load, __name__="restricted_pickle")
_stored = torch.load(MSV_CHECKPOINT, map_location="cpu", pickle_module=_restricted_pickle)
MSV_PRETRAINED_STATE = _stored["model_ema"]
MSV_IMAGENET_TOP1 = {"model": float(_stored["max_accuracy"]), "model_ema": float(_stored["max_accuray_ema"])}
del _stored
gc.collect()


def _load_module(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


msv_config_module = _load_module("msvmamba_config", MSV_DIR / "classification/config.py")
msv_models = _load_module("msvmamba_models", MSV_DIR / "classification/models/__init__.py",
                          MSV_DIR / "classification/models")
msv_utils = sys.modules["msvmamba_models.utils"]
print("MSVMamba selective-scan mode:", msv_utils.SSMODE, "(expected: mamba_ssm, the mamba-ssm 2.2.2 kernel of this stack)")
assert msv_utils.SSMODE == "mamba_ssm"


def msvmamba_config(num_classes):
    config = msv_config_module._C.clone()
    config.defrost()
    config.merge_from_file(str(MSV_DIR / CONFIG["comparator_config"]))
    config.MODEL.NUM_CLASSES = int(num_classes)
    config.freeze()
    return config


_imagenet = msv_models.build_vssm_model(msvmamba_config(1000))
_imagenet.load_state_dict(MSV_PRETRAINED_STATE, strict=True)
assert sum(p.numel() for p in _imagenet.parameters()) == 33046312
del _imagenet


class MSVMambaTiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.msvmamba = msv_models.build_vssm_model(msvmamba_config(num_classes))
        state = {key: value for key, value in MSV_PRETRAINED_STATE.items() if not key.startswith("classifier.head.")}
        result = self.msvmamba.load_state_dict(state, strict=False)
        assert sorted(result.missing_keys) == ["classifier.head.bias", "classifier.head.weight"], result
        assert not result.unexpected_keys, result
        self.feature_dim = self.msvmamba.num_features
        self.uses_consistency = False

    def parameter_groups(self):
        head = list(self.msvmamba.classifier.head.parameters())
        head_ids = {id(parameter) for parameter in head}
        return [parameter for parameter in self.parameters() if id(parameter) not in head_ids], head

    def forward(self, images):
        return empty_outputs(self.msvmamba(images))


_build_model_e34 = build_model


def build_model(name):
    if name == "msvmamba_tiny":
        return MSVMambaTiny(NUM_CLASSES)
    return _build_model_e34(name)


MSV_SOURCE = {
    "repository": MSV_REPO, "commit": MSV_COMMIT, "files_sha256": MSV_FILE_SHA256,
    "config": CONFIG["comparator_config"], "checkpoint_drive_id": MSV_CHECKPOINT_DRIVE_ID,
    "checkpoint_sha256": MSV_CHECKPOINT_SHA256, "weights": CONFIG["comparator_weights"],
    "imagenet_top1_stored_in_checkpoint": MSV_IMAGENET_TOP1, "scan_mode": msv_utils.SSMODE,
    "new_head": "nn.Linear(768, 16), MSVMamba initialisation (truncated normal std 0.02, bias 0), drawn after set_seed(seed)",
}
atomic_json(ANALYSIS_ROOT / "msvmamba_source.json", MSV_SOURCE)
print("MSVMamba-Tiny ready:", MSV_COMMIT[:12], "| weights model_ema, ImageNet top-1 stored:", MSV_IMAGENET_TOP1,
      "| parameters at 16 classes:", 32289616)


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


RECIPE_KEYS = ('epochs', 'weight_decay', 'grad_clip', 'lambda_mask',
               'mask_bce_weight', 'mask_dice_weight', 'mask_pos_weight_cap',
               'num_workers')

def run_identity(spec, seed):
    payload = {
        'extension_version': EXTENSION_VERSION,
        'seed': int(seed), 'spec': dict(spec),
        'recipe': {key: CONFIG[key] for key in RECIPE_KEYS},
        'protocol_fingerprint': PROTOCOL_FINGERPRINT,
        'data_content_fingerprint': DATA_CONTENT_FINGERPRINT,
        'class_names': CLASS_NAMES,
        'pretrained_snapshot_sha256': MODEL_SNAPSHOT_SHA256,
        'source_reference': SOURCE_REFERENCE,
        'r1_source': R1_SOURCE,
        'exif_policy': CONFIG['exif_policy'],
        'comparator_source': MSV_SOURCE,
        'core_packages': {key: RUNTIME_INFO['packages'][key] for key in
                          ('torch', 'torchvision', 'numpy', 'transformers',
                           'mamba-ssm', 'causal-conv1d', 'timm')},
    }
    return {'sha256': canonical_hash(payload), 'payload': payload}

def check_identity(blob, expected):
    actual = blob.get('run_identity', {})
    assert actual.get('sha256') == expected['sha256'], (
        'Saved run identity differs from the current configuration. '
        'Do not overwrite that run; inspect the configuration or use a new destination.'
    )
    assert canonical_hash(actual['payload']) == expected['sha256']
    assert blob['protocol_fingerprint'] == PROTOCOL_FINGERPRINT

def run_paths(spec, seed):
    name = f"cleanv1_{spec['name']}_seed{seed}"
    directory = EXTENSION_ROOT / name
    return name, directory, directory / 'best_model.pth'

RESULT_ROLES = {
    'optimization_validation': 'validation',
    'plantseg_official_test': 'plantseg_test',
    'plantdoc_external_clean': 'plantdoc_external',
}

def validate_result_summary(summary, spec, seed, expected_parameters=None):
    assert summary['model'] == spec['model']
    assert int(summary['seed']) == int(seed)
    assert summary['protocol'] == CONFIG['protocol']
    assert summary['protocol_fingerprint'] == PROTOCOL_FINGERPRINT
    assert summary.get('official_tests_evaluated') is True
    assert 1 <= int(summary['best_epoch']) <= CONFIG['epochs']
    for key, expected in spec.items():
        assert summary['spec'].get(key) == expected, (key, summary['spec'])
    assert int(summary['parameters']) == (expected_parameters or EXPECTED_PARAMETERS[spec['model']])
    for key, role in RESULT_ROLES.items():
        result = summary[key]
        frame = manifests[role]
        targets = np.asarray(result['targets'], dtype=int)
        predictions = np.asarray(result['predictions'], dtype=int)
        ids = list(result['paths'])
        assert len(targets) == len(predictions) == len(ids) == len(frame)
        assert len(set(ids)) == len(ids) and set(ids) == set(frame['record_id'])
        label_by_id = frame.set_index('record_id')['label'].to_dict()
        assert targets.tolist() == [int(label_by_id[item]) for item in ids]
        assert bool(((predictions >= 0) & (predictions < NUM_CLASSES)).all())
        recalculated = 100.0 * f1_score(targets, predictions, average='macro', zero_division=0)
        assert abs(recalculated - float(result['macro_f1'])) < 1e-6, (key, recalculated)
        for metric in ('accuracy', 'balanced_accuracy', 'macro_f1'):
            assert np.isfinite(float(result[metric])) and 0 <= float(result[metric]) <= 100
    return summary

def reference_summary(model, seed):
    spec = REFERENCE_SPECS[model]
    directory = REFERENCE_ROOT / f"cleanv1_{spec['name']}_seed{seed}"
    summary = load_json(directory / 'final_locked_summary.json')
    validate_result_summary(summary, spec, seed)
    return directory, summary

plan = {
    'extension_version': EXTENSION_VERSION, 'seeds': list(NEW_SEEDS), 'run_specs': RUN_SPECS,
    'reference_runs': {name: [f"cleanv1_{spec['name']}_seed{seed}" for seed in NEW_SEEDS]
                       for name, spec in REFERENCE_SPECS.items()},
    'reference_root': str(REFERENCE_ROOT),
    'recipe': {key: CONFIG[key] for key in RECIPE_KEYS},
    'training_epochs': CONFIG['epochs'],
    'exif_policy': CONFIG['exif_policy'],
    'comparator_source': MSV_SOURCE,
    'protocol_fingerprint': PROTOCOL_FINGERPRINT,
    'data_content_fingerprint': DATA_CONTENT_FINGERPRINT,
    'pretrained_snapshot': MODEL_SNAPSHOT,
    'source_reference': SOURCE_REFERENCE,
    'r1_source': R1_SOURCE,
}
plan_path = EXTENSION_ROOT / 'extension_plan.json'
if plan_path.exists():
    stored = load_json(plan_path)
    assert stored['plan'] == json.loads(json.dumps(plan)), 'The frozen plan differs. Inspect before continuing.'
else:
    atomic_json(plan_path, {'created_utc': utc_now(), 'plan': plan})
atomic_json(ANALYSIS_ROOT / 'runtime_last_session.json', RUNTIME_INFO)
for seed in NEW_SEEDS:
    for model in REFERENCE_SPECS:
        reference_summary(model, seed)
print('Plan saved and checked. Main training reference summaries found and valid for every seed.')


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def make_optimizer(model, spec):
    encoder_parameters, head_parameters = model.parameter_groups()
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": spec["encoder_lr"]},
            {"params": head_parameters, "lr": spec["head_lr"]},
        ],
        weight_decay=CONFIG["weight_decay"],
    )


def make_scheduler(optimizer, spec):
    warmup = spec["warmup_epochs"]
    total = CONFIG["epochs"]

    def multiplier(epoch):
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_experiment(spec):
    seed = CONFIG["seed"]
    run_name = f"cleanv1_{spec['name']}_seed{seed}"
    run_dir = Path(CONFIG["drive_runs_root"]) / run_name
    checkpoint_path = run_dir / "best_model.pth"
    last_state_path = run_dir / "last_training_state.pth"
    summary_path = run_dir / "training_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity(spec, seed)

    if summary_path.exists() and not checkpoint_path.exists():
        raise RuntimeError(f"Training summary exists without its checkpoint: {run_dir}")

    if CONFIG["skip_completed"] and checkpoint_path.exists() and summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        check_identity(summary, identity)
        assert summary["epochs_completed"] == CONFIG["epochs"]
        assert summary["checkpoint_sha256"] == sha256_path(checkpoint_path)
        print(f"Skipping completed optimization: {run_name}")
        return summary

    if checkpoint_path.exists() and not last_state_path.exists():
        raise RuntimeError(
            f"Partial checkpoint without a resume state: {run_dir}. "
            "Preserve that folder and inspect it before restarting this seed."
        )

    set_seed(seed)
    model = build_model(spec["model"]).to(DEVICE)
    optimizer = make_optimizer(model, spec)
    scheduler = make_scheduler(optimizer, spec)
    criterion = nn.CrossEntropyLoss(label_smoothing=spec["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    history = []
    best_f1 = -1.0
    best_epoch = None
    start_epoch = 0
    previous_elapsed = 0.0

    if last_state_path.exists():
        state = torch.load(last_state_path, map_location="cpu")
        check_identity(state, identity)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best_f1 = state["best_f1"]
        best_epoch = state["best_epoch"]
        start_epoch = state["epoch"]
        previous_elapsed = state.get("elapsed_seconds", 0.0)
        restore_rng_state(state["rng_state"])
        del state
        print(f"Resuming {run_name} at epoch {start_epoch + 1}/{CONFIG['epochs']}")

    validation_loader = make_validation_loader(spec)
    started = time.time()

    print("\n" + "=" * 88)
    print(
        f"Training {run_name} | size=384 | effective_batch=32 | "
        f"parameters={count_parameters(model)/1e6:.2f}M"
    )
    print("Checkpoint selection: optimization validation macro-F1 only")
    print("=" * 88)

    for epoch in range(start_epoch, CONFIG["epochs"]):
        train_loader = make_train_loader(spec, seed + epoch)
        model.train()
        running_total = running_classification = running_mask = 0.0
        correct = samples = 0
        empty_targets = 0
        mask_fraction_sum = 0.0
        finite_samples = 0
        nonfinite_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (images, masks, labels, _) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                classification_loss = criterion(outputs["logits"], labels)
                mask_loss = (
                    combined_mask_loss(outputs["mask_logits_by_scale"], masks)
                    if outputs["mask_logits_by_scale"] is not None and spec["lambda_mask"] > 0.0
                    else classification_loss.new_zeros(())
                )
                total_loss = classification_loss + spec["lambda_mask"] * mask_loss

            scaler.scale(total_loss / spec["accumulation_steps"]).backward()
            should_step = (
                (batch_index + 1) % spec["accumulation_steps"] == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            if bool(torch.isfinite(total_loss.detach())):
                running_total += total_loss.item() * batch_size
                running_classification += classification_loss.item() * batch_size
                running_mask += mask_loss.item() * batch_size
                finite_samples += batch_size
            else:
                nonfinite_batches += 1
            correct += (outputs["logits"].argmax(1) == labels).sum().item()
            samples += batch_size
            empty_targets += int((masks.flatten(1).sum(1) == 0).sum().item())
            mask_fraction_sum += float(masks.flatten(1).mean(1).sum().item())

        scheduler.step()
        validation = evaluate(
            model, validation_loader, "PlantSeg-optimization-validation", has_masks=True
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": running_total / finite_samples if finite_samples else None,
            "classification_loss": running_classification / finite_samples if finite_samples else None,
            "mask_loss": running_mask / finite_samples if finite_samples else None,
            "train_nonfinite_batches": nonfinite_batches,
            "train_accuracy": 100.0 * correct / samples,
            "train_empty_mask_targets": empty_targets,
            "train_mean_mask_fraction": mask_fraction_sum / samples,
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            **{
                f"val_{key}": value
                for key, value in validation.items()
                if key != "dataset"
            },
        }
        history.append(record)

        message = (
            f"Ep {epoch+1:02d}/{CONFIG['epochs']} "
            f"loss={(record['train_loss'] if record['train_loss'] is not None else float('nan')):.4f} "
            f"trainAcc={record['train_accuracy']:.2f} "
            f"valAcc={validation['accuracy']:.2f} "
            f"valF1={validation['macro_f1']:.2f}"
        )
        if "mask_dice" in validation:
            message += f" maskDice={validation['mask_dice']:.2f}"
        print(message)

        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_epoch = epoch + 1
            atomic_torch_save(
                {
                    "run_name": run_name,
                    "model_name": spec["model"],
                    "seed": seed,
                    "epoch": best_epoch,
                    "model_state": model.state_dict(),
                    "config": dict(CONFIG),
                    "spec": dict(spec),
                    "class_names": CLASS_NAMES,
                    "protocol_fingerprint": PROTOCOL_FINGERPRINT,
                    "run_identity": identity,
                    "runtime": RUNTIME_INFO,
                },
                checkpoint_path,
            )

        atomic_json(run_dir / "training_log.json", history)

        atomic_torch_save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "elapsed_seconds": previous_elapsed + time.time() - started,
                "protocol_fingerprint": PROTOCOL_FINGERPRINT,
                "run_identity": identity,
                "rng_state": capture_rng_state(),
                "runtime": RUNTIME_INFO,
            },
            last_state_path,
        )
        del train_loader

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    check_identity(checkpoint, identity)
    model.load_state_dict(checkpoint["model_state"])
    selected_validation = evaluate(
        model,
        validation_loader,
        "PlantSeg-optimization-validation",
        has_masks=True,
        save_predictions=False,
    )
    summary = {
        "name": spec["name"],
        "model": spec["model"],
        "seed": seed,
        "best_epoch": best_epoch,
        "parameters": count_parameters(model),
        "training_minutes": (previous_elapsed + time.time() - started) / 60.0,
        "optimization_validation": selected_validation,
        "spec": dict(spec),
        "protocol": CONFIG["protocol"],
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "official_tests_evaluated": False,
        "epochs_completed": CONFIG["epochs"],
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "run_identity": identity,
        "config": dict(CONFIG),
        "runtime": RUNTIME_INFO,
        "completed_utc": utc_now(),
    }
    atomic_json(summary_path, summary)
    if last_state_path.exists():
        last_state_path.unlink()

    print("Best validation epoch:", best_epoch)
    print_metrics(selected_validation)
    print("Official PlantSeg test and PlantDoc were NOT evaluated.")
    del model, optimizer, scheduler, scaler, checkpoint, validation_loader
    gc.collect()
    torch.cuda.empty_cache()
    return summary


reference_dir, reference = reference_summary("current_full", 11)
training_reference = load_json(reference_dir / "training_summary.json")
assert sha256_path(reference_dir / "best_model.pth") == training_reference["checkpoint_sha256"]
checkpoint = torch.load(reference_dir / "best_model.pth", map_location="cpu")
model = build_model("current_full").to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
del checkpoint
recheck = evaluate(model, make_validation_loader(REFERENCE_SPECS["current_full"]),
                   "Main training seed-11 MSLS-VMamba validation recheck", has_masks=True, save_predictions=True)
stored = reference["optimization_validation"]
assert list(recheck["paths"]) == list(stored["paths"])
agreement = float(np.mean(np.asarray(recheck["predictions"]) == np.asarray(stored["predictions"])))
environment_check = {"gpu": RUNTIME_INFO["gpu"], "reference_gpu": reference.get("runtime", {}).get("gpu"),
                     "agreement": agreement, "macro_f1_recheck": recheck["macro_f1"],
                     "macro_f1_stored": stored["macro_f1"], "checked_utc": utc_now()}
atomic_json(ANALYSIS_ROOT / "environment_check_vs_main_training.json", environment_check)
print(environment_check)
assert agreement >= 0.995, "Validation predictions differ from the main training runs: stop and inspect the environment."
del model
gc.collect()
torch.cuda.empty_cache()


from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

_tf32 = torch.backends.cuda.matmul.allow_tf32
torch.backends.cuda.matmul.allow_tf32 = False
generator = torch.Generator(device="cpu").manual_seed(0)
groups, channels, length = 4, 192, 2304
def _random(*shape):
    return torch.randn(*shape, generator=generator).to(DEVICE)
tensors = (_random(2, groups * channels, length), 0.5 * _random(2, groups * channels, length),
           -(0.5 + _random(groups * channels, 1).abs()), _random(2, groups, 1, length), _random(2, groups, 1, length),
           _random(groups * channels), 0.1 * _random(groups * channels))
weights = _random(2, groups * channels, length)

def _scan_with_gradients(function):
    leaves = [tensor.clone().requires_grad_(True) for tensor in tensors]
    output = function(*leaves)
    (output * weights).sum().backward()
    return output.detach(), [leaf.grad.detach() for leaf in leaves]

out_kernel, grads_kernel = _scan_with_gradients(lambda u, d, A, B, C, D, b: msv_utils.SelectiveScan.apply(u, d, A, B, C, D, b, True, 1))
out_ref, grads_ref = _scan_with_gradients(lambda u, d, A, B, C, D, b: selective_scan_ref(u, d, A, B, C, D, None, b, True))
torch.backends.cuda.matmul.allow_tf32 = _tf32

def _relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))

scan_check = {"output_relative_error": _relative(out_kernel, out_ref),
              "gradient_relative_error": {name: _relative(a, b) for name, a, b in
                                          zip(("u", "delta", "A", "B", "C", "D", "delta_bias"), grads_kernel, grads_ref)},
              "shapes": {"batch": 2, "groups": groups, "channels_per_group": channels, "length": length, "d_state": 1}}
print(json.dumps(scan_check, indent=1))
assert scan_check["output_relative_error"] < 1e-3 and max(scan_check["gradient_relative_error"].values()) < 5e-3, (
    "The MSVMamba scan does not match the mamba-ssm reference: stop here and inspect this output.")
del tensors, weights, out_kernel, grads_kernel, out_ref, grads_ref

spec = RUN_SPECS[0]
set_seed(11)
model = build_model(spec["model"]).to(DEVICE)
model.train()
if DEVICE == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
images = _random(spec["batch_size"], 3, spec["img_size"], spec["img_size"])
labels = torch.randint(0, NUM_CLASSES, (spec["batch_size"],), generator=generator).to(DEVICE)
started = time.time()
with torch.cuda.amp.autocast(enabled=True):
    logits = model(images)["logits"]
    loss = nn.functional.cross_entropy(logits, labels)
loss.backward()
if DEVICE == "cuda":
    torch.cuda.synchronize()
step_check = {"logits_shape": list(logits.shape), "loss": float(loss), "seconds": time.time() - started,
              "finite_gradients": bool(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())),
              "peak_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else None}
print(step_check)
assert step_check["logits_shape"] == [spec["batch_size"], NUM_CLASSES]
assert math.isfinite(step_check["loss"]) and step_check["finite_gradients"]
atomic_json(ANALYSIS_ROOT / "msvmamba_kernel_and_step_check.json", {"scan": scan_check, "step": step_check,
                                                                     "gpu": RUNTIME_INFO["gpu"], "checked_utc": utc_now()})
del model, images, labels, logits, loss
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print("Scan and training-step checks passed.")


for spec in RUN_SPECS:
    loader = make_train_loader(spec, CONFIG["seed"])
    images, masks, labels, _ = next(iter(loader))
    images = images[:2].to(DEVICE)
    labels = labels[:2].to(DEVICE)

    model = build_model(spec["model"]).to(DEVICE)
    assert count_parameters(model) == EXPECTED_PARAMETERS[spec["model"]], (
        "Unexpected parameter count. Inspect the MSVMamba checkpoint before training."
    )
    for key, value in model.msvmamba.state_dict().items():
        if not key.startswith("classifier.head."):
            assert torch.equal(value.detach().cpu(), MSV_PRETRAINED_STATE[key]), key
    encoder_parameters, head_parameters = model.parameter_groups()
    assert len(head_parameters) == 2 and len(encoder_parameters) + 2 == len(list(model.parameters()))
    model.train()
    outputs = model(images)
    assert outputs["logits"].shape == (2, NUM_CLASSES) and outputs["mask_logits_by_scale"] is None
    loss = nn.CrossEntropyLoss(label_smoothing=spec["label_smoothing"])(outputs["logits"], labels)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert bool(finite_gradients)
    assert all(parameter.grad is not None for parameter in head_parameters)
    print(spec["name"], "| input", tuple(images.shape), "| logits", tuple(outputs["logits"].shape),
          "| pretrained tensors unchanged:", len(MSV_PRETRAINED_STATE) - 2, "| labels only (lambda_mask 0)",
          "| finite gradients", bool(finite_gradients))
    del model, outputs, loss, loader, images, masks, labels
    torch.cuda.empty_cache()

print("MSVMamba-Tiny passed the forward and backward test on real training images.")


import portalocker

TRAIN_SEEDS = NEW_SEEDS

free_gb = shutil.disk_usage(STORAGE).free / 1e9
print(f'free storage space: {free_gb:.1f} GB (5 checkpoints need about 0.7 GB, plus 0.4 GB while a run is resumable)')
assert free_gb >= 2.0, 'Not enough free storage space: free some space first.'

training_summaries = []
with portalocker.Lock(SCRATCH + '/vmamba_msls_execution.lock', timeout=1):
    for seed in TRAIN_SEEDS:
        CONFIG['seed'] = int(seed)
        for spec in RUN_SPECS:
            print(f"\nSeed {seed}: {spec['name']}")
            training_summaries.append(train_experiment(spec))

assert len(training_summaries) == len(TRAIN_SEEDS) * len(RUN_SPECS)
assert all(summary['epochs_completed'] == CONFIG['epochs'] for summary in training_summaries)
assert {(summary['model'], int(summary['seed'])) for summary in training_summaries} == {
    (spec['model'], seed) for spec in RUN_SPECS for seed in TRAIN_SEEDS
}
print(f'{len(training_summaries)} MSVMamba-Tiny run(s) completed. No test set selected a checkpoint.')


_untrained = [int(seed) for seed in NEW_SEEDS for spec in RUN_SPECS
              if not (run_paths(spec, seed)[1] / 'training_summary.json').exists()]
assert not _untrained, f'Seeds {_untrained} are not trained yet. No test set has been read.'

def verify_all_new_training_complete():
    for seed in NEW_SEEDS:
        for spec in RUN_SPECS:
            _, directory, checkpoint = run_paths(spec, seed)
            summary = load_json(directory / 'training_summary.json')
            check_identity(summary, run_identity(spec, seed))
            assert summary['epochs_completed'] == CONFIG['epochs']
            assert checkpoint.exists() and sha256_path(checkpoint) == summary['checkpoint_sha256']

def evaluate_new_run(spec, seed):
    CONFIG['seed'] = int(seed)
    run_name, directory, checkpoint_path = run_paths(spec, seed)
    identity = run_identity(spec, seed)
    training = load_json(directory / 'training_summary.json')
    check_identity(training, identity)
    checkpoint_sha = sha256_path(checkpoint_path)
    assert checkpoint_sha == training['checkpoint_sha256']
    final_path = directory / 'final_locked_summary.json'
    if final_path.exists():
        summary = load_json(final_path)
        check_identity(summary, identity)
        assert summary['checkpoint_sha256'] == checkpoint_sha
        assert summary['best_epoch'] == training['best_epoch']
        validate_result_summary(summary, spec, seed)
        print('Reusing locked evaluation:', run_name)
        return summary
    set_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    check_identity(checkpoint, identity)
    assert checkpoint['seed'] == seed and checkpoint['model_name'] == spec['model']
    assert checkpoint['class_names'] == CLASS_NAMES
    assert checkpoint['epoch'] == training['best_epoch']
    model = build_model(spec['model']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    del checkpoint
    results = {
        'optimization_validation': evaluate(
            model, make_validation_loader(spec), 'PlantSeg-optimization-validation',
            has_masks=True, save_predictions=True)
    }
    loaders = make_final_evaluation_loaders(spec)
    for name, (loader, has_masks) in loaders.items():
        results[name] = evaluate(model, loader, name, has_masks=has_masks,
                                 save_predictions=True)
    summary = {
        'name': spec['name'], 'model': spec['model'], 'seed': int(seed),
        'best_epoch': training['best_epoch'], 'parameters': count_parameters(model),
        'spec': dict(spec), 'config': dict(CONFIG), 'protocol': CONFIG['protocol'],
        'protocol_fingerprint': PROTOCOL_FINGERPRINT,
        'official_tests_evaluated': True, 'epochs_completed': CONFIG['epochs'],
        'checkpoint_sha256': checkpoint_sha, 'run_identity': identity,
        'runtime': RUNTIME_INFO, 'evaluated_utc': utc_now(), **results,
    }
    validate_result_summary(summary, spec, seed)
    atomic_json(final_path, summary)
    print('LOCKED:', run_name)
    for result in results.values():
        print_metrics(result)
    del model, loaders, loader
    gc.collect()
    torch.cuda.empty_cache()
    return summary

new_final_summaries = []
with portalocker.Lock(SCRATCH + '/vmamba_msls_execution.lock', timeout=1):
    verify_all_new_training_complete()
    for seed in NEW_SEEDS:
        for spec in RUN_SPECS:
            new_final_summaries.append(evaluate_new_run(spec, seed))
assert len(new_final_summaries) == len(NEW_SEEDS) * len(RUN_SPECS)
print('All five final evaluations completed or verified and reused.')

access = []
for summary in new_final_summaries:
    for split in ('plantseg_official_test', 'plantdoc_external_clean'):
        access.append({'run': f"cleanv1_{summary['name']}_seed{summary['seed']}", 'split': split,
                       'checkpoint_sha256': summary['checkpoint_sha256'],
                       'evaluated_utc': summary.get('evaluated_utc'), 'purpose': 'MSVMamba-Tiny locked evaluation'})
atomic_json(EXTENSION_ROOT / 'test_access_log.json', access)
print(len(access), 'test accesses logged (one per checkpoint and test split).')


from scipy.stats import t as student_t

def paired_statistics(differences):
    differences = np.asarray(differences, dtype=float)
    assert differences.ndim == 1 and len(differences) >= 2
    assert np.isfinite(differences).all()
    n = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    standard_error = sd / math.sqrt(n)
    margin = float(student_t.ppf(0.975, n - 1)) * standard_error
    return {'n_pairs': n, 'mean_gain': mean, 'sd_gain': sd,
            'standard_error': standard_error,
            'ci95_lower': mean - margin, 'ci95_upper': mean + margin,
            'positive_seeds': int((differences > 0).sum()),
            'ties': int((differences == 0).sum())}

summaries = {}
for spec in RUN_SPECS:
    for seed in NEW_SEEDS:
        _, directory, _ = run_paths(spec, seed)
        summary = load_json(directory / 'final_locked_summary.json')
        check_identity(summary, run_identity(spec, seed))
        validate_result_summary(summary, spec, seed)
        summaries[(spec['name'], seed)] = (directory / 'final_locked_summary.json', summary)
for model, spec in REFERENCE_SPECS.items():
    for seed in NEW_SEEDS:
        directory, summary = reference_summary(model, seed)
        summaries[(spec['name'], seed)] = (directory / 'final_locked_summary.json', summary)

rows = []
for (name, seed), (path, summary) in summaries.items():
    for dataset in RESULT_ROLES:
        result = summary[dataset]
        row = {'run_name': name, 'seed': int(seed), 'model': summary['model'], 'dataset': dataset,
               'source': 'MSVMamba-Tiny comparison' if path.parent.parent == EXTENSION_ROOT else 'Main training (read only)',
               'best_epoch': int(summary['best_epoch']), 'parameters_M': summary['parameters'] / 1e6,
               'summary_path': str(path), 'summary_sha256': sha256_path(path)}
        for metric in ('accuracy', 'balanced_accuracy', 'macro_f1', 'mask_dice', 'mask_iou'):
            row[metric] = result.get(metric, np.nan)
        rows.append(row)
per_seed = pd.DataFrame(rows).sort_values(['run_name', 'seed', 'dataset'])
assert not per_seed.duplicated(['run_name', 'seed', 'dataset']).any()
per_seed.to_csv(ANALYSIS_ROOT / 'per_seed_results.csv', index=False)

mean_rows = []
for (name, dataset), group in per_seed.groupby(['run_name', 'dataset']):
    for metric in ('accuracy', 'balanced_accuracy', 'macro_f1'):
        values = group[metric].to_numpy(dtype=float)
        mean_rows.append({'run_name': name, 'dataset': dataset, 'metric': metric, 'n_seeds': len(values),
                          'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1))})
model_means = pd.DataFrame(mean_rows)
model_means.to_csv(ANALYSIS_ROOT / 'model_mean_sample_sd.csv', index=False)

MSLS, VMAMBA = REFERENCE_SPECS['current_full']['name'], REFERENCE_SPECS['vmamba_baseline']['name']
COMPARISONS = [(MSLS, spec['name']) for spec in RUN_SPECS] + [(spec['name'], VMAMBA) for spec in RUN_SPECS]
effect_rows, gain_rows = [], []
for left, right in COMPARISONS:
    for dataset in RESULT_ROLES:
        for metric in ('macro_f1', 'accuracy', 'balanced_accuracy'):
            pivot = per_seed[per_seed['dataset'] == dataset].pivot(index='seed', columns='run_name', values=metric).reindex(NEW_SEEDS)
            differences = (pivot[left] - pivot[right]).to_numpy(dtype=float)
            effect_rows.append({'comparison': f'{left} - {right}', 'dataset': dataset, 'metric': metric,
                                **paired_statistics(differences)})
            if metric == 'macro_f1':
                for seed, value in zip(NEW_SEEDS, differences):
                    gain_rows.append({'comparison': f'{left} - {right}', 'dataset': dataset, 'seed': int(seed), 'difference': float(value)})
paired_effects = pd.DataFrame(effect_rows)
paired_effects.to_csv(ANALYSIS_ROOT / 'paired_effects_ci95.csv', index=False)
pd.DataFrame(gain_rows).to_csv(ANALYSIS_ROOT / 'paired_differences_by_seed.csv', index=False)

display(model_means[model_means.metric == 'macro_f1'].pivot(index='run_name', columns='dataset', values='mean').round(2))
display(paired_effects[paired_effects.metric == 'macro_f1'].round(3))

for spec in RUN_SPECS:
    for seed in NEW_SEEDS:
        _, summary = summaries[(spec['name'], seed)]
        target_dir = ANALYSIS_ROOT / 'per_class' / f"{spec['name']}_seed{seed}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for dataset in ('plantseg_official_test', 'plantdoc_external_clean'):
            result = summary[dataset]
            report = classification_report(result['targets'], result['predictions'], labels=np.arange(NUM_CLASSES),
                                           target_names=CLASS_NAMES, output_dict=True, zero_division=0)
            pd.DataFrame(report).transpose().to_csv(target_dir / f'{dataset}_classification_report.csv')
            matrix = confusion_matrix(result['targets'], result['predictions'], labels=np.arange(NUM_CLASSES))
            pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(target_dir / f'{dataset}_confusion_matrix.csv')

diagnostic_rows = []
for spec in RUN_SPECS:
    for seed in NEW_SEEDS:
        _, directory, _ = run_paths(spec, seed)
        for record in load_json(directory / 'training_log.json'):
            diagnostic_rows.append({'run_name': spec['name'], 'seed': seed, 'epoch': record['epoch'],
                                    'train_empty_mask_targets': record.get('train_empty_mask_targets'),
                                    'train_mean_mask_fraction': record.get('train_mean_mask_fraction')})
pd.DataFrame(diagnostic_rows).to_csv(ANALYSIS_ROOT / 'training_mask_diagnostics.csv', index=False)
print('Saved per-seed results, means, paired effects, per-class reports and mask diagnostics to', ANALYSIS_ROOT)
