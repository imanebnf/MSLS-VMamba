# Lesion perturbation on the test set, five seeds; fills runs/lesion_perturbation_seeds11-55.
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
    "drive_runs_root": STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/lesion_perturbation_11_55_v1",
    "reference_runs_root": STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1",
    "exif_policy": "original_uncorrected",
    "blur_radius_px": 6,
    "exg_threshold": 20,
    "leaf_fraction_min": 0.80,
    "lesion_margin_px": 3,
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
EXTENSION_VERSION = "lesion_perturbation_11_55_v1"
EXPECTED_PARAMETERS = {"vmamba_baseline": 29492368, "current_full": 30405907,
                       "current_full_uniform_map": 30405907}
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
print("No training: the 10 main training checkpoints are read only.")
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


EXIF_ORIENTATION_TAG = 0x0112
TO_RAW = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270,
          2: Image.Transpose.FLIP_LEFT_RIGHT, 4: Image.Transpose.FLIP_TOP_BOTTOM,
          5: Image.Transpose.TRANSPOSE, 7: Image.Transpose.TRANSVERSE}
test_rows = []
with zipfile.ZipFile(plantseg_zip) as archive:
    members = set(archive.namelist())
    for row in manifests["plantseg_test"].itertuples(index=False):
        reference = re.sub(r"images/test/(.*)\.jpg$", r"annotations/test/\1.png", row.image_reference)
        assert reference != row.image_reference and reference in members, (row.image_reference, reference)
        target = Path(CONFIG["local_cache_root"]) / "plantseg_corrected_test_masks" / Path(reference).name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read(reference)
        target.write_bytes(data)
        with Image.open(row.local_image) as image:
            raw_size = image.size
            orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
        test_rows.append({"record_id": row.record_id, "label": int(row.label), "harmonized_class": row.harmonized_class,
                          "local_image": row.local_image, "corrected_mask": str(target),
                          "corrected_mask_reference": reference, "mask_sha256": hashlib.sha256(data).hexdigest(),
                          "exif_orientation": orientation, "raw_width": raw_size[0], "raw_height": raw_size[1]})
test_table = pd.DataFrame(test_rows)
assert test_table.corrected_mask_reference.is_unique and len(test_table) == len(manifests["plantseg_test"])
print("Corrected test masks:", len(test_table), "| EXIF-tagged test images:", int((test_table.exif_orientation != 1).sum()))

def lesion_on_raw_grid(row):
    with Image.open(row.corrected_mask) as mask_file:
        mask = mask_file.convert("L")
    realigned = False
    if row.exif_orientation in TO_RAW:
        mask = mask.transpose(TO_RAW[row.exif_orientation])
        realigned = True
    return mask, realigned, mask.size == (row.raw_width, row.raw_height)


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


from scipy.signal import correlate
from scipy.ndimage import binary_dilation, distance_transform_edt
from PIL import ImageFilter

SIZE = RUN_SPECS[0]["img_size"]
FILL = (124, 116, 104)

def content_box(width, height, size):
    scale = float(size) / float(max(width, height))
    new_width, new_height = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    box = np.zeros((size, size), dtype=bool)
    box[top:top + new_height, left:left + new_width] = True
    return box

region_rows, inputs = [], {}
for row in test_table.itertuples(index=False):
    with Image.open(row.local_image) as image_file:
        image = resize_and_pad(image_file.convert("RGB"), SIZE, InterpolationMode.BICUBIC, FILL)
    mask_raw, realigned, size_ok = lesion_on_raw_grid(row)
    lesion = (np.asarray(resize_and_pad(mask_raw, SIZE, InterpolationMode.NEAREST, 0)) > 0)
    content = content_box(row.raw_width, row.raw_height, SIZE)
    pixels = np.asarray(image).astype(np.int32)
    green = ((2 * pixels[..., 1] - pixels[..., 0] - pixels[..., 2]) > CONFIG["exg_threshold"]) & content & ~lesion
    margin = CONFIG["lesion_margin_px"]
    forbidden = binary_dilation(lesion, structure=np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool)) | ~content
    area = int(lesion.sum())
    flags = []
    if realigned:
        flags.append("exif_mask_realigned")
    if not size_ok:
        flags.append("mask_size_mismatch")
    region, method, green_fraction, distance = None, "none", np.nan, np.nan
    fallback = None
    if area == 0:
        flags.append("no_lesion")
    else:
        rows_, cols_ = np.nonzero(lesion)
        top, left = rows_.min(), cols_.min()
        crop = lesion[top:rows_.max() + 1, left:cols_.max() + 1].astype(np.float32)
        hit = np.rint(correlate(forbidden.astype(np.float32), crop, mode="valid", method="fft"))
        tissue = np.rint(correlate(green.astype(np.float32), crop, mode="valid", method="fft"))
        valid = hit <= 0
        if valid.any():
            yy, xx = np.indices(valid.shape)
            score = np.where(valid, tissue, -1.0)
            best = score.max()
            candidates = np.argwhere(valid & (score == best))
            distances = np.hypot(candidates[:, 0] - top, candidates[:, 1] - left)
            y0, x0 = candidates[int(np.argmin(distances))]
            region = np.zeros_like(lesion)
            region[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop > 0
            assert int(region.sum()) == area and not (region & forbidden).any()
            method, green_fraction, distance = "translated_copy", float((region & green).sum()) / area, float(distances.min())
            if green_fraction < CONFIG["leaf_fraction_min"]:
                flags.append("leaf_not_verified")
        else:
            flags.append("no_translated_region")
        candidates_pixels = green & ~forbidden
        if candidates_pixels.sum() >= area:
            distance_map = distance_transform_edt(~lesion)
            order = np.argsort(np.where(candidates_pixels, distance_map, np.inf), axis=None, kind="stable")[:area]
            fallback = np.zeros(lesion.size, dtype=bool)
            fallback[order] = True
            fallback = fallback.reshape(lesion.shape)
        else:
            flags.append("insufficient_tissue")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=CONFIG["blur_radius_px"]))
    def composite(mask):
        return np.asarray(Image.composite(blurred, image, Image.fromarray(mask.astype(np.uint8) * 255, mode="L")))
    inputs[row.record_id] = {
        "original": np.asarray(image),
        "lesion_blurred": composite(lesion) if area else None,
        "region_blurred": composite(region) if region is not None else None,
        "fallback_blurred": composite(fallback) if fallback is not None else None,
    }
    primary = (area > 0 and region is not None and "leaf_not_verified" not in flags and "mask_size_mismatch" not in flags)
    region_rows.append({"record_id": row.record_id, "label": row.label, "harmonized_class": row.harmonized_class,
                        "lesion_pixels_384": area, "lesion_fraction_of_content": area / max(1, int(content.sum())),
                        "region_method": method, "region_green_fraction": green_fraction,
                        "region_offset_px": distance, "fallback_available": fallback is not None,
                        "primary_set": bool(primary), "flags": ";".join(flags),
                        "exif_orientation": row.exif_orientation})
regions = pd.DataFrame(region_rows)
(EXTENSION_ROOT / "regions").mkdir(parents=True, exist_ok=True)
regions.to_csv(EXTENSION_ROOT / "regions" / "plantseg_test_regions.csv", index=False)
print("Test images:", len(regions), "| primary set (translated copy verified on green tissue):", int(regions.primary_set.sum()))
display(regions["flags"].str.get_dummies(sep=";").sum().rename("images").to_frame())


def to_tensor_batch(arrays):
    return torch.stack([TF.normalize(TF.to_tensor(Image.fromarray(a)), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        for a in arrays])

@torch.no_grad()
def probabilities(model, arrays, batch_size=32):
    out = []
    for start in range(0, len(arrays), batch_size):
        batch = to_tensor_batch(arrays[start:start + batch_size]).to(DEVICE)
        out.append(torch.softmax(model(batch)["logits"].float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, NUM_CLASSES))

ids = list(regions.record_id)
labels = regions.set_index("record_id").label.to_dict()
summary_rows, access = [], []
for spec in RUN_SPECS:
    for seed in NEW_SEEDS:
        run_name = f"cleanv1_{spec['name']}_seed{seed}"
        directory = REFERENCE_ROOT / run_name
        training = load_json(directory / "training_summary.json")
        locked = load_json(directory / "final_locked_summary.json")
        checkpoint_sha = sha256_path(directory / "best_model.pth")
        assert checkpoint_sha == training["checkpoint_sha256"] == locked["checkpoint_sha256"], run_name
        checkpoint = torch.load(directory / "best_model.pth", map_location="cpu")
        model = build_model(spec["model"]).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        del checkpoint
        stored = dict(zip(locked["plantseg_official_test"]["paths"], locked["plantseg_official_test"]["predictions"]))
        p_original = probabilities(model, [inputs[i]["original"] for i in ids])
        agreement = float(np.mean([int(np.argmax(p)) == int(stored[i]) for i, p in zip(ids, p_original)]))
        assert agreement >= 0.995, (run_name, agreement)
        out = {"record_id": ids, "label": [labels[i] for i in ids],
               "pred_original": p_original.argmax(1), "p_true_original": p_original[np.arange(len(ids)), [labels[i] for i in ids]]}
        for kind in ("lesion_blurred", "region_blurred", "fallback_blurred"):
            subset = [i for i in ids if inputs[i][kind] is not None]
            probs = dict(zip(subset, probabilities(model, [inputs[i][kind] for i in subset])))
            out[f"pred_{kind}"] = [int(probs[i].argmax()) if i in probs else -1 for i in ids]
            out[f"p_true_{kind}"] = [float(probs[i][labels[i]]) if i in probs else np.nan for i in ids]
        frame = pd.DataFrame(out).merge(regions[["record_id", "primary_set", "flags", "region_method",
                                                 "region_green_fraction", "lesion_fraction_of_content",
                                                 "fallback_available"]], on="record_id")
        target = EXTENSION_ROOT / f"{spec['name']}_seed{seed}"
        target.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target / "per_image_perturbation.csv", index=False)
        for subset_name, subset in (("primary", frame[frame.primary_set]),
                                    ("fallback", frame[frame.fallback_available & (frame.pred_fallback_blurred >= 0)])):
            if len(subset) == 0:
                continue
            second = "region_blurred" if subset_name == "primary" else "fallback_blurred"
            row = {"run_name": run_name, "model": spec["model"], "seed": seed, "subset": subset_name, "n_images": len(subset),
                   "unperturbed_agreement_with_locked": agreement}
            for kind in ("original", "lesion_blurred", second):
                predictions = subset[f"pred_{kind}"].to_numpy()
                row[f"accuracy_{kind}"] = 100.0 * float(np.mean(predictions == subset.label.to_numpy()))
                row[f"macro_f1_{kind}"] = 100.0 * f1_score(subset.label, predictions, average="macro", zero_division=0)
                row[f"mean_p_true_{kind}"] = float(subset[f"p_true_{kind}"].mean())
                if kind != "original":
                    row[f"flip_rate_{kind}"] = float(np.mean(predictions != subset.pred_original.to_numpy()))
            row["p_true_drop_lesion_minus_region"] = float((subset.p_true_original - subset.p_true_lesion_blurred).mean()
                                                          - (subset.p_true_original - subset[f"p_true_{second}"]).mean())
            summary_rows.append(row)
        access.append({"run": run_name, "split": "plantseg_official_test", "checkpoint_sha256": checkpoint_sha,
                       "evaluated_utc": utc_now(), "purpose": "perturbation analysis (no selection)"})
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(run_name, "| agreement with locked predictions", round(agreement, 4))
perturbation_summary = pd.DataFrame(summary_rows)
perturbation_summary.to_csv(EXTENSION_ROOT / "perturbation_summary.csv", index=False)
atomic_json(EXTENSION_ROOT / "test_access_log.json", access)
display(perturbation_summary.round(3))
