# Seed-11 ablation variant: plain multitask; fills runs/seed11_ablation_training.
import os

STORAGE = os.environ.get("MSLS_STORAGE", "storage")
SCRATCH = os.environ.get("MSLS_SCRATCH", "scratch")


def display(obj):
    print(obj)


import hashlib
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageClassification

import matplotlib.pyplot as plt
import seaborn as sns
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

warnings.filterwarnings("ignore")

CONFIG = {
    "protocol": "clean_protocol_v1",
    "control_version": "plain_multitask_control_seed11_v1",
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
    "class_mapping": STORAGE + "/msls_vmamba/data/class_mapping.json",
    "reference_runs_root": STORAGE + "/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1",
    "drive_runs_root": STORAGE + "/msls_vmamba/runs/seed11_analyses/plain_multitask_control_seed11_v1",
    "local_cache_root": SCRATCH + "/vmamba_clean_protocol_v1_multitask_control",
    "model_repo": "saurabhati/VMamba_ImageNet_82.6",
    "model_dir": SCRATCH + "/vmamba_model",
}

RUN_SPECS = [
    {
        "name": "vmamba_plain_multitask_384",
        "model": "vmamba_plain_multitask",
        "img_size": 384,
        "batch_size": 16,
        "accumulation_steps": 2,
        "label_smoothing": 0.10,
        "warmup_epochs": 4,
        "encoder_lr": 5e-5,
        "head_lr": 5e-4,
        "mask_target_mode": "correct",
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


set_seed(CONFIG["seed"])
Path(CONFIG["drive_runs_root"]).mkdir(parents=True, exist_ok=True)

print("torch:", torch.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VMamba CUDA kernels: OK")
display(pd.DataFrame(RUN_SPECS))


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
mapping_path = Path(CONFIG["class_mapping"])

for required in (manifest_root, plantseg_zip, mapping_path):
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
    assert len(frame) == EXPECTED_COUNTS[role], (role, len(frame), EXPECTED_COUNTS[role])
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

for key in ("duplicate_cluster", "sha256"):
    train_values = set(manifests["train"][key])
    validation_values = set(manifests["validation"][key])
    test_values = set(manifests["plantseg_test"][key])
    external_values = set(manifests["plantdoc_external"][key])
    assert train_values.isdisjoint(validation_values)
    assert train_values.isdisjoint(test_values)
    assert validation_values.isdisjoint(test_values)
    assert (train_values | validation_values | test_values).isdisjoint(external_values)

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

CROP_NAMES = [name.split("_", 1)[0] for name in CLASS_NAMES]
crop_to_index = {name: index for index, name in enumerate(sorted(set(CROP_NAMES)))}
CLASS_TO_CROP = np.asarray([crop_to_index[name] for name in CROP_NAMES], dtype=int)
SAME_CROP_NEGATIVE = (
    CLASS_TO_CROP[:, None] == CLASS_TO_CROP[None, :]
) & (~np.eye(NUM_CLASSES, dtype=bool))

print("Protocol:", CONFIG["protocol"])
print("Fingerprint:", PROTOCOL_FINGERPRINT)
print("Train/validation:", len(manifests["train"]), len(manifests["validation"]))
print("Official-test and external manifests were validated but their images will not be loaded.")
display(pd.DataFrame({"class": CLASS_NAMES, "crop": CROP_NAMES}))


local_root = Path(CONFIG["local_cache_root"])
local_ps_root = local_root / "plantseg"
local_ps_root.mkdir(parents=True, exist_ok=True)


def sha256_path(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


optimization_rows = pd.concat(
    [manifests["train"], manifests["validation"]], ignore_index=True
).drop_duplicates("record_id").reset_index(drop=True)
local_paths = {}

with zipfile.ZipFile(plantseg_zip) as archive:
    archive_members = set(archive.namelist())
    for number, row in enumerate(optimization_rows.itertuples(index=False), start=1):
        assert row.dataset == "PlantSeg"
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

        if not mask_path.exists():
            with archive.open(row.mask_reference) as source, mask_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        with Image.open(mask_path) as mask:
            mask.verify()

        local_paths[row.record_id] = (str(image_path), str(mask_path))
        if number % 500 == 0:
            print(f"Verified local cache {number}/{len(optimization_rows)}")

for role in ("train", "validation"):
    frame = manifests[role].copy()
    frame["local_image"] = frame["record_id"].map(lambda value: local_paths[value][0])
    frame["local_mask"] = frame["record_id"].map(lambda value: local_paths[value][1])
    manifests[role] = frame

print(f"Verified {len(optimization_rows)} optimization images and masks.")
print("No official-test or external image was opened.")


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
    def __init__(self, frame, size, train, mask_target_mode="correct",
                 mask_permutation_seed=42):
        self.frame = frame.reset_index(drop=True)
        self.transform = (
            PairedStrongTrainTransform(size)
            if train else PairedEvalTransform(size)
        )
        if mask_target_mode not in ("correct", "same_class_permuted"):
            raise ValueError(f"Unknown mask target mode: {mask_target_mode}")
        if not train and mask_target_mode != "correct":
            raise ValueError("Evaluation masks must stay correctly paired")

        labels = self.frame["label"].to_numpy(dtype=int)
        self.mask_donor_indices = np.arange(len(self.frame))
        if mask_target_mode == "same_class_permuted":
            rng = np.random.default_rng(mask_permutation_seed)
            for class_index in range(NUM_CLASSES):
                positions = np.flatnonzero(labels == class_index)
                assert len(positions) >= 2, (
                    "Each shuffled-mask class requires two training images",
                    class_index,
                )
                shuffled = rng.permutation(positions)
                self.mask_donor_indices[shuffled] = np.roll(shuffled, 1)
            assert np.all(self.mask_donor_indices != np.arange(len(self.frame)))
            assert np.array_equal(labels, labels[self.mask_donor_indices])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        donor_row = self.frame.iloc[self.mask_donor_indices[index]]
        with Image.open(row.local_image) as image_file:
            image = image_file.convert("RGB")
        with Image.open(donor_row.local_mask) as mask_file:
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
        manifests["train"], spec["img_size"], train=True,
        mask_target_mode=spec["mask_target_mode"],
        mask_permutation_seed=CONFIG["seed"] + 314159,
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


display(pd.DataFrame({
    "class": CLASS_NAMES,
    "optimization_train": class_counts,
}))
print("Validation loaders are used for checkpoint selection.")
print("No PlantSeg official-test or PlantDoc loader is created in this script.")


MODEL_DIR = CONFIG["model_dir"]
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    snapshot_download(repo_id=CONFIG["model_repo"], local_dir=MODEL_DIR,
                      revision="3db342a1b00a91c350dca4df93a1645ad2ac9493")
import hashlib as _hl
assert _hl.sha256(open(os.path.join(MODEL_DIR, "model.safetensors"), "rb").read()).hexdigest() == (
    "45e94dd6155b492baff6203b1099c9a6b1082b17e2b640246aa1896311d228e1"), "Pretrained VMamba weights differ from the main training snapshot; use a clean CONFIG[model_dir]."


def load_pretrained_vmamba():
    return AutoModelForImageClassification.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )


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
            lesion_vectors.append(self.weighted_pool(projected, probability))
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


def build_model(name):
    if name == "vmamba_plain_multitask":
        return PlainMultitaskVMamba(NUM_CLASSES)
    raise ValueError(f"Unknown model: {name}")


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

    if CONFIG["skip_completed"] and checkpoint_path.exists() and summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        assert summary["protocol_fingerprint"] == PROTOCOL_FINGERPRINT
        print(f"Skipping completed optimization: {run_name}")
        return summary

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
        state = torch.load(last_state_path, map_location=DEVICE)
        assert state["protocol_fingerprint"] == PROTOCOL_FINGERPRINT
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best_f1 = state["best_f1"]
        best_epoch = state["best_epoch"]
        start_epoch = state["epoch"]
        previous_elapsed = state.get("elapsed_seconds", 0.0)
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
            torch.save(
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
                },
                checkpoint_path,
            )

        with (run_dir / "training_log.json").open("w") as handle:
            json.dump(history, handle, indent=2)

        torch.save(
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
            },
            last_state_path,
        )
        del train_loader

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
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
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    if last_state_path.exists():
        last_state_path.unlink()

    print("Best validation epoch:", best_epoch)
    print_metrics(selected_validation)
    print("Official PlantSeg test and PlantDoc were NOT evaluated.")
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return summary


assert len(RUN_SPECS) == 1
spec = RUN_SPECS[0]

loader = make_train_loader(spec, CONFIG["seed"])
images, masks, labels, _ = next(iter(loader))
images = images[:2].to(DEVICE)
masks = masks[:2].to(DEVICE)
labels = labels[:2].to(DEVICE)

model = build_model(spec["model"]).to(DEVICE)
model.train()
outputs = model(images)

assert outputs["logits"].shape == (2, NUM_CLASSES)
assert set(outputs["mask_logits_by_scale"]) == {"fine", "mid", "coarse"}

classification_loss = nn.CrossEntropyLoss(
    label_smoothing=spec["label_smoothing"]
)(outputs["logits"], labels)
mask_loss = combined_mask_loss(outputs["mask_logits_by_scale"], masks)
loss = classification_loss + CONFIG["lambda_mask"] * mask_loss
loss.backward()

finite_gradients = all(
    parameter.grad is None or torch.isfinite(parameter.grad).all()
    for parameter in model.parameters()
)
assert bool(finite_gradients)

classification_gradient_reaches_backbone = any(
    parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
    for parameter in model.vmamba.parameters()
)
assert classification_gradient_reaches_backbone

print("Input:", tuple(images.shape))
print("Logits:", tuple(outputs["logits"].shape))
print("Masks:", {
    key: tuple(value.shape)
    for key, value in outputs["mask_logits_by_scale"].items()
})
print("Finite gradients:", bool(finite_gradients))
print("Smoke test: PASS")

del model, outputs, loss, loader
torch.cuda.empty_cache()


training_summaries = [train_experiment(RUN_SPECS[0])]
assert len(training_summaries) == 1
assert training_summaries[0]["protocol_fingerprint"] == PROTOCOL_FINGERPRINT

print("\nPlain multi-task control finished optimization.")
print("No official test set was used during training or checkpoint selection.")
