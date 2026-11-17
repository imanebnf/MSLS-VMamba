# Trains ResNet50, EfficientNetV2-S and ViT-S/16 with five seeds; fills runs/baselines_seeds11-55.
import os

STORAGE = os.environ.get("MSLS_STORAGE", "storage")
SCRATCH = os.environ.get("MSLS_SCRATCH", "scratch")


def display(obj):
    print(obj)


import torch
import numpy as np

import hashlib, gc, itertools, json, math, os, platform, random, shutil, sys, time, uuid, warnings, zipfile
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode, ColorJitter
import torchvision
import torchvision.models as tvm
import timm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from huggingface_hub import HfApi, hf_hub_download
import matplotlib.pyplot as plt
import seaborn as sns

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
    "drive_runs_root": STORAGE + "/msls_vmamba/runs/baselines_seeds11_55_v1",
    "local_cache_root": SCRATCH + "/vmamba_clean_protocol_v1",
}
SEEDS = (11, 22, 33, 44, 55)
EXPERIMENT_VERSION = "baselines_seeds11_55_v1"
EXPECTED_PROTOCOL_FINGERPRINT = "ff6e6b3bca40e614062e611a50d6736690a3658d433593f500a0c34c651ec648"
EXPECTED_DATA_CONTENT_FINGERPRINT = "5c340193e4b10f1a04cec57254013d448eca32e3622729d063ce8b464aad4a05"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

BASELINES = {
    "resnet50": {"library": "torchvision", "constructor": "resnet50", "weights": "ResNet50_Weights.IMAGENET1K_V2",
                 "head": "fc", "label": "ResNet50"},
    "efficientnet_v2_s": {"library": "torchvision", "constructor": "efficientnet_v2_s",
                          "weights": "EfficientNet_V2_S_Weights.IMAGENET1K_V1", "head": "classifier.1",
                          "label": "EfficientNetV2-S"},
    "vit_s16": {"library": "timm", "constructor": "vit_small_patch16_224.augreg_in1k", "weights": "timm pretrained tag augreg_in1k",
                "head": "head", "img_size": 384, "label": "ViT-S/16"},
}
MODEL_ORDER = ["resnet50", "efficientnet_v2_s", "vit_s16"]
RUN_SPECS = [{"name": f"{key}_384", "model": key, "img_size": 384, "batch_size": 16, "accumulation_steps": 2,
              "label_smoothing": 0.10, "warmup_epochs": 4, "encoder_lr": 5e-5, "head_lr": 5e-4} for key in MODEL_ORDER]

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


set_seed(CONFIG["seed"])
Path(CONFIG["drive_runs_root"]).mkdir(parents=True, exist_ok=True)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


RUNTIME_INFO = {
    "python": sys.version, "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    "packages": {name: package_version(name) for name in (
        "torch", "torchvision", "numpy", "timm", "huggingface-hub", "safetensors", "scikit-learn", "scipy", "pandas")},
}
for name, expected in {"torch": "2.2.2", "torchvision": "0.17.2", "timm": "0.9.16", "numpy": "1.26.4"}.items():
    actual = RUNTIME_INFO["packages"][name]
    assert actual and actual.split("+")[0] == expected, f"{name}: expected {expected}, got {actual}."
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__,
      "| GPU", RUNTIME_INFO["gpu"])
display(pd.DataFrame(RUN_SPECS))


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


assert DATA_CONTENT_FINGERPRINT == EXPECTED_DATA_CONTENT_FINGERPRINT, (
    "Data content differs from the main training runs. Stop and inspect the sources.")
print("Data content fingerprint matches the main training VMamba / MSLS-VMamba runs.")


def mask_encoding_summary(frame):
    value_counts, fractions_gt0, fractions_gt127, sizes, modes = {}, [], [], {}, {}
    for path in frame["local_mask"]:
        with Image.open(path) as handle:
            modes[handle.mode] = modes.get(handle.mode, 0) + 1
            sizes[str(handle.size)] = sizes.get(str(handle.size), 0) + 1
            array = np.asarray(handle.convert("L"))
        values, counts = np.unique(array, return_counts=True)
        for v, c in zip(values.tolist(), counts.tolist()):
            value_counts[v] = value_counts.get(v, 0) + c
        fractions_gt0.append(float((array > 0).mean()))
        fractions_gt127.append(float((array > 127).mean()))
    total = sum(value_counts.values())
    top = sorted(value_counts.items(), key=lambda kv: -kv[1])[:12]
    q = lambda xs: [float(np.percentile(xs, p)) for p in (5, 25, 50, 75, 95)]
    return {
        "n_masks": int(len(frame)), "pil_modes": modes, "n_distinct_values": len(value_counts),
        "top_values_share": {str(v): c / total for v, c in top},
        "share_pixels_value_1_to_127": sum(c for v, c in value_counts.items() if 0 < v <= 127) / total,
        "lesion_fraction_gt0_mean": float(np.mean(fractions_gt0)), "lesion_fraction_gt0_quantiles_5_25_50_75_95": q(fractions_gt0),
        "lesion_fraction_gt127_mean": float(np.mean(fractions_gt127)), "lesion_fraction_gt127_quantiles_5_25_50_75_95": q(fractions_gt127),
        "images_with_empty_mask_gt0": int(sum(f == 0 for f in fractions_gt0)),
        "distinct_sizes": len(sizes),
    }

mask_check = {role: mask_encoding_summary(manifests[role]) for role in ("train", "validation", "plantseg_test")}
mask_check["note"] = "Training binarizes with > 0. Read-only diagnostic for OPEN_QUESTIONS I24."
atomic_json(EXTENSION_ROOT / "mask_encoding_check.json", mask_check)
display(pd.DataFrame({k: {m: v for m, v in s.items() if not isinstance(v, (dict, list))} for k, s in mask_check.items() if k != "note"}))


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


def empty_outputs(logits):
    return {"logits": logits, "mask_logits": None, "mask_logits_by_scale": None, "soft_mask": None, "fusion_gates": None}


def torchvision_weights(name):
    enum_name, member = name.split(".")
    return getattr(getattr(tvm, enum_name), member)


def get_submodule(module, dotted):
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_submodule(module, dotted, new):
    parts = dotted.split(".")
    parent = get_submodule(module, ".".join(parts[:-1])) if len(parts) > 1 else module
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new
    else:
        setattr(parent, parts[-1], new)


class BaselineClassifier(nn.Module):
    def __init__(self, key, num_classes, pretrained=True):
        super().__init__()
        cfg = BASELINES[key]
        self.key, self.head_name = key, cfg["head"]
        if cfg["library"] == "timm":
            self.net = timm.create_model(cfg["constructor"], pretrained=pretrained, num_classes=num_classes,
                                         img_size=cfg["img_size"])
            mean, std = self.net.pretrained_cfg["mean"], self.net.pretrained_cfg["std"]
        else:
            weights = torchvision_weights(cfg["weights"])
            self.net = getattr(tvm, cfg["constructor"])(weights=weights if pretrained else None)
            old = get_submodule(self.net, self.head_name)
            set_submodule(self.net, self.head_name, nn.Linear(old.in_features, num_classes))
            mean, std = weights.transforms().mean, weights.transforms().std
        self.input_mean, self.input_std = tuple(float(v) for v in mean), tuple(float(v) for v in std)
        in_mean, in_std = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        mean_t, std_t = torch.tensor(self.input_mean), torch.tensor(self.input_std)
        self.register_buffer("renorm_scale", (in_std / std_t).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("renorm_shift", ((in_mean - mean_t) / std_t).view(1, 3, 1, 1), persistent=False)
        self.uses_consistency = False

    def head_module(self):
        return get_submodule(self.net, self.head_name)

    def parameter_groups(self):
        head = list(self.head_module().parameters())
        head_ids = {id(p) for p in head}
        encoder = [p for p in self.parameters() if id(p) not in head_ids]
        return encoder, head

    def forward(self, images):
        return empty_outputs(self.net(images * self.renorm_scale + self.renorm_shift))


def build_model(name, pretrained=True):
    if name in BASELINES:
        return BaselineClassifier(name, NUM_CLASSES, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


def pretrained_weight_record(key):
    cfg = BASELINES[key]
    if cfg["library"] == "timm":
        hub_id = timm.create_model(cfg["constructor"], pretrained=False).pretrained_cfg.get("hf_hub_id")
        filename = "model.safetensors"
        try:
            path = hf_hub_download(hub_id, filename)
        except Exception:
            filename = "pytorch_model.bin"
            path = hf_hub_download(hub_id, filename)
        revision = HfApi().model_info(hub_id).sha
        return {"library": "timm", "identifier": cfg["constructor"], "hf_hub_id": hub_id, "hf_revision": revision,
                "file": filename, "sha256": sha256_path(path)}
    weights = torchvision_weights(cfg["weights"])
    path = Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(weights.url)
    if not path.exists():
        getattr(tvm, cfg["constructor"])(weights=weights)
    return {"library": "torchvision", "identifier": f"torchvision.models.{cfg['constructor']}", "weights": cfg["weights"],
            "url": weights.url, "imagenet_acc1": weights.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1"),
            "sha256": sha256_path(path)}


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


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def lr_multiplier(epoch, warmup, total):
    if epoch < warmup:
        return float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def preflight(spec):
    key = spec["model"]
    cfg = BASELINES[key]
    report = {"model": key, "definition": cfg}
    set_seed(1234)
    model = build_model(key).to(DEVICE)
    head = model.head_module()
    report["head_module"] = cfg["head"]
    report["head_out_features"] = int(head.out_features)
    assert head.out_features == NUM_CLASSES == 16
    report["parameters"] = count_parameters(model)
    report["input_mean_std"] = {"mean": model.input_mean, "std": model.input_std}

    loader = make_train_loader(spec, 1234)
    images, masks, labels, _ = next(iter(loader))
    del loader
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(images[:4])["logits"]
    assert tuple(logits.shape) == (4, NUM_CLASSES) and torch.isfinite(logits).all()
    report["logits_shape"] = list(logits.shape)

    ours = {k: v.detach().cpu() for k, v in model.net.state_dict().items()}
    if cfg["library"] == "timm":
        reference = timm.create_model(cfg["constructor"], pretrained=True).state_dict()
        from timm.layers import resample_abs_pos_embed
        grid = cfg["img_size"] // 16
        expected_pos = resample_abs_pos_embed(reference["pos_embed"], new_size=(grid, grid), num_prefix_tokens=1)
        report["pos_embed_shapes"] = {"pretrained": list(reference["pos_embed"].shape), "ours": list(ours["pos_embed"].shape)}
        report["pos_embed_max_abs_diff_vs_timm_resample"] = float((ours["pos_embed"] - expected_pos).abs().max())
        assert report["pos_embed_max_abs_diff_vs_timm_resample"] < 1e-5, "Positional embedding not resampled as expected"
        assert torch.equal(ours["pos_embed"][:, :1], reference["pos_embed"][:, :1]), "class-token position changed"
        skip = ("head.", "pos_embed")
    else:
        reference = getattr(tvm, cfg["constructor"])(weights=torchvision_weights(cfg["weights"])).state_dict()
        skip = (cfg["head"] + ".",)
    compared = mismatched = 0
    for name, tensor in ours.items():
        if name.startswith(skip) or name == "pos_embed":
            continue
        compared += 1
        if name not in reference or not torch.equal(tensor, reference[name].cpu()):
            mismatched += 1
    missing_from_ours = [k for k in reference if not k.startswith(skip) and k != "pos_embed" and k not in ours]
    report["pretrained_tensors_compared"], report["pretrained_tensors_mismatched"] = compared, mismatched
    report["pretrained_keys_missing_from_model"] = missing_from_ours
    assert mismatched == 0 and not missing_from_ours, "Pretrained weights were not loaded intact"
    del reference

    encoder, head_params = model.parameter_groups()
    enc_ids, head_ids = {id(p) for p in encoder}, {id(p) for p in head_params}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert not enc_ids & head_ids and (enc_ids | head_ids) == trainable
    names = {id(p): n for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": spec["encoder_lr"]}, {"params": head_params, "lr": spec["head_lr"]}],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_multiplier(e, spec["warmup_epochs"], CONFIG["epochs"]))
    report["lr_groups"] = {"encoder": {"n_tensors": len(encoder), "n_params": int(sum(p.numel() for p in encoder)),
                                       "initial_lr": optimizer.param_groups[0]["initial_lr"]},
                           "head": {"n_tensors": len(head_params), "n_params": int(sum(p.numel() for p in head_params)),
                                    "initial_lr": optimizer.param_groups[1]["initial_lr"],
                                    "names": [names[id(p)] for p in head_params]}}
    assert optimizer.param_groups[0]["initial_lr"] == spec["encoder_lr"] and optimizer.param_groups[1]["initial_lr"] == spec["head_lr"]
    report["lr_multiplier_epochs_0_1_3_4_20_39"] = [lr_multiplier(e, spec["warmup_epochs"], CONFIG["epochs"]) for e in (0, 1, 3, 4, 20, 39)]

    criterion = nn.CrossEntropyLoss(label_smoothing=spec["label_smoothing"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for group, lr in zip(optimizer.param_groups, (spec["encoder_lr"], spec["head_lr"])):
        group["lr"] = lr
    model.train()
    losses = []
    for step in range(12):
        with torch.cuda.amp.autocast(enabled=True):
            loss = criterion(model(images)["logits"], labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
    scheduler.step()
    report["dry_run_losses"] = losses
    assert all(np.isfinite(losses)) and np.mean(losses[-3:]) < np.mean(losses[:3]), "Loss did not decrease on a fixed batch"

    model.eval()
    probe = torch.randn(2, 3, spec["img_size"], spec["img_size"], generator=torch.Generator().manual_seed(7)).to(DEVICE)
    with torch.no_grad():
        before = model(probe)["logits"].float()
    path = Path(SCRATCH + "/preflight_roundtrip.pth")
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict()}, path)
    restored = build_model(key, pretrained=False).to(DEVICE)
    blob = torch.load(path, map_location=DEVICE)
    restored.load_state_dict(blob["model_state"])
    restored.eval()
    with torch.no_grad():
        after = restored(probe)["logits"].float()
    report["checkpoint_roundtrip_max_abs_logit_diff"] = float((before - after).abs().max())
    assert report["checkpoint_roundtrip_max_abs_logit_diff"] <= 1e-5
    enc2, head2 = restored.parameter_groups()
    opt2 = torch.optim.AdamW([{"params": enc2, "lr": spec["encoder_lr"]}, {"params": head2, "lr": spec["head_lr"]}],
                             weight_decay=CONFIG["weight_decay"])
    sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda e: lr_multiplier(e, spec["warmup_epochs"], CONFIG["epochs"]))
    sc2 = torch.cuda.amp.GradScaler(enabled=True)
    opt2.load_state_dict(blob["optimizer_state"]); sch2.load_state_dict(blob["scheduler_state"]); sc2.load_state_dict(blob["scaler_state"])
    exp_avg_equal = all(torch.equal(a["exp_avg"], b["exp_avg"]) for a, b in zip(optimizer.state_dict()["state"].values(),
                                                                                 opt2.state_dict()["state"].values()))
    report["optimizer_state_roundtrip_equal"] = bool(exp_avg_equal)
    report["scheduler_last_epoch_restored"] = sch2.last_epoch == scheduler.last_epoch
    report["scaler_scale_restored"] = sc2.get_scale() == scaler.get_scale()
    assert exp_avg_equal and report["scheduler_last_epoch_restored"] and report["scaler_scale_restored"]
    path.unlink()

    report["pretrained_weights"] = pretrained_weight_record(key)
    report["passed"] = True
    del model, restored, optimizer, opt2, blob, images, labels, masks
    gc.collect(); torch.cuda.empty_cache()
    return report


PREFLIGHT_PATH = EXTENSION_ROOT / "baseline_preflight.json"
preflight_reports = {}
for spec in RUN_SPECS:
    print("Preflight:", spec["model"])
    preflight_reports[spec["model"]] = preflight(spec)
    r = preflight_reports[spec["model"]]
    print(f"  params={r['parameters']:,} head={r['head_module']}->{r['head_out_features']} "
          f"tensors compared={r['pretrained_tensors_compared']} mismatched={r['pretrained_tensors_mismatched']} "
          f"loss {r['dry_run_losses'][0]:.3f}->{r['dry_run_losses'][-1]:.3f} roundtrip diff={r['checkpoint_roundtrip_max_abs_logit_diff']}")
preflight_payload = {"created_utc": utc_now(), "runtime": RUNTIME_INFO, "reports": preflight_reports,
                     "all_passed": all(r["passed"] for r in preflight_reports.values())}
atomic_json(PREFLIGHT_PATH, preflight_payload)
assert preflight_payload["all_passed"]
WEIGHT_RECORDS = {k: r["pretrained_weights"] for k, r in preflight_reports.items()}
EXPECTED_PARAMETERS = {k: r["parameters"] for k, r in preflight_reports.items()}
print("All baselines passed preflight. Saved", PREFLIGHT_PATH)


RECIPE_KEYS = ("epochs", "weight_decay", "grad_clip", "num_workers")


def run_identity(spec, seed):
    payload = {
        "experiment_version": EXPERIMENT_VERSION, "seed": int(seed), "spec": dict(spec),
        "recipe": {key: CONFIG[key] for key in RECIPE_KEYS},
        "protocol_fingerprint": PROTOCOL_FINGERPRINT, "data_content_fingerprint": DATA_CONTENT_FINGERPRINT,
        "class_names": CLASS_NAMES, "baseline_definition": BASELINES[spec["model"]],
        "input_normalization": preflight_reports[spec["model"]]["input_mean_std"],
        "pretrained_weights": WEIGHT_RECORDS[spec["model"]], "source_reference": SOURCE_REFERENCE,
        "core_packages": {key: RUNTIME_INFO["packages"][key] for key in ("torch", "torchvision", "numpy", "timm")},
    }
    return {"sha256": canonical_hash(payload), "payload": payload}


def check_identity(blob, expected):
    actual = blob.get("run_identity", {})
    assert actual.get("sha256") == expected["sha256"], (
        "Saved run identity differs from the current configuration. Do not overwrite that run; inspect it.")
    assert canonical_hash(actual["payload"]) == expected["sha256"]
    assert blob["protocol_fingerprint"] == PROTOCOL_FINGERPRINT


def run_paths(spec, seed):
    name = f"cleanv1_{spec['name']}_seed{seed}"
    directory = EXTENSION_ROOT / name
    return name, directory, directory / "best_model.pth"


RESULT_ROLES = {"optimization_validation": "validation", "plantseg_official_test": "plantseg_test",
                "plantdoc_external_clean": "plantdoc_external"}


def validate_result_summary(summary, spec, seed):
    assert summary["model"] == spec["model"] and int(summary["seed"]) == int(seed)
    assert summary["protocol_fingerprint"] == PROTOCOL_FINGERPRINT and summary.get("official_tests_evaluated") is True
    assert 1 <= int(summary["best_epoch"]) <= CONFIG["epochs"]
    assert int(summary["parameters"]) == EXPECTED_PARAMETERS[spec["model"]]
    for key, role in RESULT_ROLES.items():
        result, frame = summary[key], manifests[role]
        targets = np.asarray(result["targets"], dtype=int)
        predictions = np.asarray(result["predictions"], dtype=int)
        ids = list(result["paths"])
        assert len(targets) == len(predictions) == len(ids) == len(frame)
        assert len(set(ids)) == len(ids) and set(ids) == set(frame["record_id"])
        label_by_id = frame.set_index("record_id")["label"].to_dict()
        assert targets.tolist() == [int(label_by_id[item]) for item in ids]
        recalculated = 100.0 * f1_score(targets, predictions, average="macro", zero_division=0)
        assert abs(recalculated - float(result["macro_f1"])) < 1e-6
        probabilities = np.asarray(result["probabilities"])
        assert probabilities.shape == (len(ids), NUM_CLASSES)
    return summary


plan = {"experiment_version": EXPERIMENT_VERSION, "seeds": list(SEEDS), "run_specs": RUN_SPECS, "baselines": BASELINES,
        "recipe": {key: CONFIG[key] for key in RECIPE_KEYS}, "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "data_content_fingerprint": DATA_CONTENT_FINGERPRINT, "pretrained_weights": WEIGHT_RECORDS,
        "source_reference": SOURCE_REFERENCE,
}
plan_path = EXTENSION_ROOT / "experiment_plan.json"
if plan_path.exists():
    assert load_json(plan_path)["plan"] == plan, "The frozen plan differs. Inspect before continuing."
else:
    atomic_json(plan_path, {"created_utc": utc_now(), "plan": plan})
atomic_json(ANALYSIS_ROOT / "runtime_last_session.json", RUNTIME_INFO)
print("Plan frozen:", plan_path)


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
                    if outputs["mask_logits_by_scale"] is not None
                    else classification_loss.new_zeros(())
                )
                total_loss = classification_loss + CONFIG["lambda_mask"] * mask_loss

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
            running_total += total_loss.item() * batch_size
            running_classification += classification_loss.item() * batch_size
            running_mask += mask_loss.item() * batch_size
            correct += (outputs["logits"].argmax(1) == labels).sum().item()
            samples += batch_size

        scheduler.step()
        validation = evaluate(
            model, validation_loader, "PlantSeg-optimization-validation", has_masks=True
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": running_total / samples,
            "classification_loss": running_classification / samples,
            "mask_loss": running_mask / samples,
            "train_accuracy": 100.0 * correct / samples,
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
            f"loss={record['train_loss']:.4f} "
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


ACCESS_LOG = EXTENSION_ROOT / "test_access_log.json"


@torch.no_grad()
def softmax_probabilities(model, loader):
    model.eval()
    probabilities, paths = [], []
    for batch in loader:
        images = batch[0].to(DEVICE, non_blocking=True)
        probabilities.append(torch.softmax(model(images)["logits"].float(), dim=1).cpu())
        paths.extend(batch[-1])
    return torch.cat(probabilities).numpy(), paths


def evaluate_with_probabilities(model, loader, name, has_masks):
    result = evaluate(model, loader, name, has_masks=has_masks, save_predictions=True)
    probabilities, paths = softmax_probabilities(model, loader)
    assert list(paths) == list(result["paths"])
    agreement = float((probabilities.argmax(1) == np.asarray(result["predictions"])).mean())
    assert agreement == 1.0, f"{name}: probability argmax disagrees with predictions ({agreement})"
    result["probabilities"] = np.round(probabilities, 6).tolist()
    return result


def locked_evaluate_model(spec):
    for seed in SEEDS:
        _, directory, checkpoint = run_paths(spec, seed)
        CONFIG["seed"] = int(seed)
        training = load_json(directory / "training_summary.json")
        check_identity(training, run_identity(spec, seed))
        assert training["epochs_completed"] == CONFIG["epochs"]
        assert checkpoint.exists() and sha256_path(checkpoint) == training["checkpoint_sha256"]
    summaries = []
    for seed in SEEDS:
        CONFIG["seed"] = int(seed)
        run_name, directory, checkpoint_path = run_paths(spec, seed)
        identity = run_identity(spec, seed)
        training = load_json(directory / "training_summary.json")
        final_path = directory / "final_locked_summary.json"
        checkpoint_sha = sha256_path(checkpoint_path)
        if final_path.exists():
            summary = load_json(final_path)
            check_identity(summary, identity)
            assert summary["checkpoint_sha256"] == checkpoint_sha
            validate_result_summary(summary, spec, seed)
            print("Reusing locked evaluation:", run_name)
            summaries.append(summary)
            continue
        set_seed(seed)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        check_identity(checkpoint, identity)
        assert checkpoint["epoch"] == training["best_epoch"] and checkpoint["class_names"] == CLASS_NAMES
        model = build_model(spec["model"], pretrained=False).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint
        results = {"optimization_validation": evaluate_with_probabilities(
            model, make_validation_loader(spec), "PlantSeg-optimization-validation", True)}
        access = load_json(ACCESS_LOG) if ACCESS_LOG.exists() else []
        for name, (loader, has_masks) in make_final_evaluation_loaders(spec).items():
            results[name] = evaluate_with_probabilities(model, loader, name, has_masks)
            access.append({"utc": utc_now(), "model": spec["model"], "seed": int(seed), "split": name,
                           "n_images": len(results[name]["targets"]), "checkpoint_sha256": checkpoint_sha})
        atomic_json(ACCESS_LOG, access)
        summary = {"name": spec["name"], "model": spec["model"], "seed": int(seed), "best_epoch": training["best_epoch"],
                   "parameters": count_parameters(model), "spec": dict(spec), "config": dict(CONFIG),
                   "protocol": CONFIG["protocol"], "protocol_fingerprint": PROTOCOL_FINGERPRINT,
                   "official_tests_evaluated": True, "epochs_completed": CONFIG["epochs"],
                   "checkpoint_sha256": checkpoint_sha, "run_identity": identity, "runtime": RUNTIME_INFO,
                   "training_minutes": training.get("training_minutes"), "evaluated_utc": utc_now(), **results}
        validate_result_summary(summary, spec, seed)
        atomic_json(final_path, summary)
        print("LOCKED:", run_name)
        for result in results.values():
            print_metrics(result)
        summaries.append(summary)
        del model
        gc.collect(); torch.cuda.empty_cache()
    return summaries


import portalocker

all_final = {}
with portalocker.Lock(SCRATCH + "/baselines_execution.lock", timeout=1):
    assert load_json(PREFLIGHT_PATH)["all_passed"]
    for spec in RUN_SPECS:
        for seed in SEEDS:
            CONFIG["seed"] = int(seed)
            print(f"\n{spec['model']} seed {seed}")
            train_experiment(spec)
        all_final[spec["model"]] = locked_evaluate_model(spec)
print("All fifteen baseline runs trained and evaluated.")


rows = []
for model_key, summaries in all_final.items():
    for summary in summaries:
        for key in RESULT_ROLES:
            r = summary[key]
            rows.append({"model": model_key, "seed": summary["seed"], "split": key, "best_epoch": summary["best_epoch"],
                         "parameters": summary["parameters"], "accuracy": r["accuracy"],
                         "balanced_accuracy": r["balanced_accuracy"], "macro_f1": r["macro_f1"],
                         "training_minutes": summary.get("training_minutes")})
per_seed = pd.DataFrame(rows)
per_seed.to_csv(ANALYSIS_ROOT / "baselines_per_seed.csv", index=False)
summary_table = per_seed.groupby(["split", "model"])[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
summary_table.to_csv(ANALYSIS_ROOT / "baselines_mean_sample_sd.csv")
display(summary_table.round(2))
print("Saved", ANALYSIS_ROOT)
