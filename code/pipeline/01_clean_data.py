# Duplicate and label-conflict audit of PlantSeg and PlantDoc; writes the partition manifests and the exclusion list (data/manifests, data/exclusions, data/cleaning_protocol.json).
import os

STORAGE = os.environ.get("MSLS_STORAGE", "storage")
SCRATCH = os.environ.get("MSLS_SCRATCH", "scratch")


def display(obj):
    print(obj)


import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.model_selection import train_test_split

SEED = 42
PHASH_THRESHOLD = 5
VALIDATION_FRACTION = 0.20

ORIGINAL_PS_ZIP = Path(STORAGE + "/plantseg.zip")
ORIGINAL_PD_ROOT = Path(STORAGE + "/PlantDoc-Dataset")
CLASS_MAPPING_PATH = Path(
    STORAGE + "/msls_vmamba/data/class_mapping.json"
)
OUTPUT_ROOT = Path(
    STORAGE + "/msls_vmamba/data/clean_protocol_v1"
)
REVIEW_ROOT = OUTPUT_ROOT / "duplicate_review"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
REVIEW_ROOT.mkdir(parents=True, exist_ok=True)

for required in (ORIGINAL_PS_ZIP, ORIGINAL_PD_ROOT, CLASS_MAPPING_PATH):
    assert required.exists(), f"Missing required source: {required}"

with open(CLASS_MAPPING_PATH, encoding="utf-8") as file:
    existing_mapping = json.load(file)

EXPECTED_CLASSES = list(existing_mapping["class_names"])
assert len(EXPECTED_CLASSES) == 16

print("Original PlantSeg:", ORIGINAL_PS_ZIP)
print("Original PlantDoc:", ORIGINAL_PD_ROOT)
print("Manifest output:", OUTPUT_ROOT)
print("Classes:", len(EXPECTED_CLASSES))


def normalize_label(value):
    value = unicodedata.normalize("NFKC", str(value))
    value = value.strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", value)


PLANTSEG_MAPPING = {
    ("apple", "apple rust"): "apple_rust",
    ("apple", "apple scab"): "apple_scab",
    ("corn", "corn gray leaf spot"): "corn_gray_leaf_spot",
    ("corn", "corn northern leaf blight"): "corn_northern_blight",
    ("corn", "corn rust"): "corn_rust",
    ("grape", "grape black rot"): "grape_black_rot",
    ("potato", "potato early blight"): "potato_early_blight",
    ("potato", "potato late blight"): "potato_late_blight",
    ("squash", "squash powdery mildew"): "squash_powdery_mildew",
    ("tomato", "tomato bacterial leaf spot"): "tomato_bacterial_spot",
    ("tomato", "tomato early blight"): "tomato_early_blight",
    ("tomato", "tomato late blight"): "tomato_late_blight",
    ("tomato", "tomato leaf mold"): "tomato_leaf_mold",
    ("tomato", "tomato mosaic virus"): "tomato_mosaic_virus",
    ("tomato", "tomato septoria leaf spot"): "tomato_septoria",
    ("tomato", "tomato yellow leaf curl virus"): "tomato_yellow_curl",
}


PLANTDOC_MAPPING = {
    "apple scab leaf": "apple_scab",
    "apple rust leaf": "apple_rust",
    "corn gray leaf spot": "corn_gray_leaf_spot",
    "corn leaf blight": "corn_northern_blight",
    "corn rust leaf": "corn_rust",
    "grape leaf black rot": "grape_black_rot",
    "potato leaf early blight": "potato_early_blight",
    "potato leaf late blight": "potato_late_blight",
    "squash powdery mildew leaf": "squash_powdery_mildew",
    "tomato leaf bacterial spot": "tomato_bacterial_spot",
    "tomato early blight leaf": "tomato_early_blight",
    "tomato leaf late blight": "tomato_late_blight",
    "tomato mold leaf": "tomato_leaf_mold",
    "tomato leaf mosaic virus": "tomato_mosaic_virus",
    "tomato septoria leaf spot": "tomato_septoria",
    "tomato leaf yellow virus": "tomato_yellow_curl",
}

assert set(PLANTSEG_MAPPING.values()) == set(EXPECTED_CLASSES)
assert set(PLANTDOC_MAPPING.values()) == set(EXPECTED_CLASSES)

mapping_audit = pd.DataFrame([
    {
        "dataset": "PlantSeg",
        "original_plant": plant,
        "original_disease": disease,
        "harmonized_class": harmonized,
    }
    for (plant, disease), harmonized in PLANTSEG_MAPPING.items()
] + [
    {
        "dataset": "PlantDoc",
        "original_plant": None,
        "original_disease": original,
        "harmonized_class": harmonized,
    }
    for original, harmonized in PLANTDOC_MAPPING.items()
])

display(mapping_audit.sort_values(["harmonized_class", "dataset"]))
mapping_audit.to_csv(OUTPUT_ROOT / "audited_class_mapping.csv", index=False)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def member_split(member, marker):
    parts = Path(member).parts
    if marker not in parts:
        return None
    position = parts.index(marker)
    if position + 1 >= len(parts):
        return None
    value = normalize_label(parts[position + 1])
    if value in {"train", "training"}:
        return "training"
    if value in {"test", "testing"}:
        return "test"
    return value


with zipfile.ZipFile(ORIGINAL_PS_ZIP) as archive:
    members = archive.namelist()
    metadata_member = next(
        member for member in members
        if member.lower().endswith("metadata.csv")
    )
    with archive.open(metadata_member) as file:
        metadata = pd.read_csv(file)

    image_members_by_name = defaultdict(list)
    mask_members_by_name = defaultdict(list)
    for member in members:
        suffix = Path(member).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            continue
        if "/images/" in member:
            image_members_by_name[Path(member).name].append(member)
        elif any(marker in member for marker in ("/annotations/", "/masks/")):
            mask_members_by_name[Path(member).name].append(member)

metadata["plant_normalized"] = metadata["Plant"].map(normalize_label)
metadata["disease_normalized"] = metadata["Disease"].map(normalize_label)
metadata["source_split"] = metadata["Training/Test"].map(normalize_label).map({
    "training": "training",
    "train": "training",
    "test": "test",
    "testing": "test",
})
metadata["mapping_key"] = list(zip(
    metadata["plant_normalized"], metadata["disease_normalized"]
))
metadata["harmonized_class"] = metadata["mapping_key"].map(PLANTSEG_MAPPING)
metadata["label_file"] = metadata["Label file"].astype(str)
selected_metadata = metadata[metadata["harmonized_class"].notna()].copy()

plantseg_records = []
for row in selected_metadata.itertuples(index=False):
    image_candidates = image_members_by_name[str(row.Name)]
    label_name = row.label_file
    mask_candidates = mask_members_by_name[str(label_name)]

    matching_images = [
        member for member in image_candidates
        if member_split(member, "images") == row.source_split
    ]
    matching_masks = [
        member for member in mask_candidates
        if member_split(member, "annotations") == row.source_split
        or member_split(member, "masks") == row.source_split
    ]

    assert len(matching_images) == 1, (
        row.Name, row.source_split, matching_images
    )
    assert len(matching_masks) == 1, (
        label_name, row.source_split, matching_masks
    )

    plantseg_records.append({
        "record_id": f"PS::{row.source_split}::{row.Name}",
        "dataset": "PlantSeg",
        "source_split": row.source_split,
        "harmonized_class": row.harmonized_class,
        "original_class": f"{row.Plant} | {row.Disease}",
        "storage": "zip",
        "source_container": str(ORIGINAL_PS_ZIP),
        "image_reference": matching_images[0],
        "mask_reference": matching_masks[0],
        "original_name": str(row.Name),
    })

plantdoc_records = []
for source_split in ("train", "test"):
    split_root = ORIGINAL_PD_ROOT / source_split
    for class_directory in sorted(split_root.iterdir()):
        if not class_directory.is_dir():
            continue
        original_normalized = normalize_label(class_directory.name)
        harmonized = PLANTDOC_MAPPING.get(original_normalized)
        if harmonized is None:
            continue
        for image_path in sorted(class_directory.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(ORIGINAL_PD_ROOT)
            plantdoc_records.append({
                "record_id": f"PD::{relative.as_posix()}",
                "dataset": "PlantDoc",
                "source_split": source_split,
                "harmonized_class": harmonized,
                "original_class": class_directory.name,
                "storage": "file",
                "source_container": str(ORIGINAL_PD_ROOT),
                "image_reference": str(image_path),
                "mask_reference": None,
                "original_name": image_path.name,
            })

records = pd.DataFrame(plantseg_records + plantdoc_records)
assert records["record_id"].is_unique
assert len(plantseg_records) == 2259, len(plantseg_records)
assert len(plantdoc_records) == 1655, len(plantdoc_records)

source_counts = (
    records.groupby(["dataset", "source_split", "harmonized_class"])
    .size().reset_index(name="images")
)
display(source_counts)
source_counts.to_csv(OUTPUT_ROOT / "original_source_counts.csv", index=False)
print("PlantSeg selected:", len(plantseg_records))
print("PlantDoc selected:", len(plantdoc_records))


def perceptual_hash_from_bytes(data, hash_size=8, highfreq_factor=4):
    image_size = hash_size * highfreq_factor
    with Image.open(io.BytesIO(data)) as image:
        gray = ImageOps.grayscale(image).resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(gray, dtype=np.float32)
    frequency = dctn(pixels, type=2, norm="ortho")
    low_frequency = frequency[:hash_size, :hash_size]
    median = np.median(low_frequency.ravel()[1:])
    bits = (low_frequency > median).ravel()
    return np.packbits(bits)


sha256_values = []
phash_values = []

with zipfile.ZipFile(ORIGINAL_PS_ZIP) as archive:
    for number, row in enumerate(records.itertuples(index=False), start=1):
        if row.storage == "zip":
            data = archive.read(row.image_reference)
        else:
            data = Path(row.image_reference).read_bytes()
        sha256_values.append(hashlib.sha256(data).hexdigest())
        phash_values.append(perceptual_hash_from_bytes(data))
        if number % 500 == 0:
            print(f"Hashed {number}/{len(records)} images")

records["sha256"] = sha256_values
phash_matrix = np.stack(phash_values).astype(np.uint8)
records["phash_hex"] = [value.tobytes().hex() for value in phash_matrix]
records.to_csv(OUTPUT_ROOT / "all_original_records_with_hashes.csv", index=False)
print("Hashing complete:", len(records))


BIT_COUNT = np.array([int(value).bit_count() for value in range(256)], dtype=np.uint8)


def append_pair(output, left_index, right_index, relation, distance):
    left = records.iloc[left_index]
    right = records.iloc[right_index]
    output.append({
        "relation": relation,
        "phash_distance": int(distance),
        "byte_identical": left.sha256 == right.sha256,
        "same_harmonized_class": (
            left.harmonized_class == right.harmonized_class
        ),
        "left_index": int(left_index),
        "left_record_id": left.record_id,
        "left_dataset": left.dataset,
        "left_source_split": left.source_split,
        "left_class": left.harmonized_class,
        "left_reference": left.image_reference,
        "right_index": int(right_index),
        "right_record_id": right.record_id,
        "right_dataset": right.dataset,
        "right_source_split": right.source_split,
        "right_class": right.harmonized_class,
        "right_reference": right.image_reference,
    })


def find_internal_pairs(record_indices, relation, threshold=PHASH_THRESHOLD, block=128):
    record_indices = np.asarray(record_indices, dtype=int)
    hashes = phash_matrix[record_indices]
    output = []
    for start in range(0, len(record_indices), block):
        left_hashes = hashes[start:start + block]
        xor = np.bitwise_xor(left_hashes[:, None, :], hashes[None, :, :])
        distances = BIT_COUNT[xor].sum(axis=2)
        for local_row in range(len(left_hashes)):
            source_position = start + local_row
            matches = np.flatnonzero(distances[local_row] <= threshold)
            matches = matches[matches > source_position]
            for target_position in matches:
                append_pair(
                    output,
                    record_indices[source_position],
                    record_indices[target_position],
                    relation,
                    distances[local_row, target_position],
                )
    return output


def find_cross_pairs(left_indices, right_indices, relation,
                     threshold=PHASH_THRESHOLD, block=128):
    left_indices = np.asarray(left_indices, dtype=int)
    right_indices = np.asarray(right_indices, dtype=int)
    right_hashes = phash_matrix[right_indices]
    output = []
    for start in range(0, len(left_indices), block):
        source_indices = left_indices[start:start + block]
        left_hashes = phash_matrix[source_indices]
        xor = np.bitwise_xor(left_hashes[:, None, :], right_hashes[None, :, :])
        distances = BIT_COUNT[xor].sum(axis=2)
        for local_row, source_index in enumerate(source_indices):
            for target_position in np.flatnonzero(
                distances[local_row] <= threshold
            ):
                append_pair(
                    output,
                    source_index,
                    right_indices[target_position],
                    relation,
                    distances[local_row, target_position],
                )
    return output


ps_indices = records.index[records["dataset"] == "PlantSeg"].to_numpy()
pd_indices = records.index[records["dataset"] == "PlantDoc"].to_numpy()

pair_rows = []
pair_rows.extend(find_internal_pairs(ps_indices, "PlantSeg internal"))
pair_rows.extend(find_internal_pairs(pd_indices, "PlantDoc internal"))
pair_rows.extend(find_cross_pairs(ps_indices, pd_indices, "PlantSeg vs PlantDoc"))

pairs = pd.DataFrame(pair_rows)
if not pairs.empty:
    pairs = pairs.sort_values([
        "byte_identical", "phash_distance", "relation"
    ], ascending=[False, True, True]).reset_index(drop=True)

display(
    pairs.groupby(
        ["relation", "byte_identical", "same_harmonized_class"],
        dropna=False,
    ).size().reset_index(name="pairs")
)
pairs.to_csv(OUTPUT_ROOT / "all_duplicate_candidates.csv", index=False)
print("Total strict candidate pairs:", len(pairs))


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


clusters = UnionFind(len(records))
for row in pairs.itertuples(index=False):
    clusters.union(row.left_index, row.right_index)

records["duplicate_cluster"] = [
    clusters.find(index) for index in range(len(records))
]
records["keep"] = False
records["exclusion_reason"] = ""


def preferred_index(frame, prefer_test=True):
    ordered = frame.copy()
    ordered["split_priority"] = ordered["source_split"].map(
        {"test": 0, "training": 1, "train": 1}
    ).fillna(2)
    ordered = ordered.sort_values(
        ["split_priority", "record_id"], kind="stable"
    )
    return int(ordered.index[0])


for cluster_id, group in records.groupby("duplicate_cluster"):
    ps_group = group[group["dataset"] == "PlantSeg"]
    pd_group = group[group["dataset"] == "PlantDoc"]

    if not ps_group.empty:
        ps_classes = set(ps_group["harmonized_class"])
        if len(ps_classes) > 1:
            records.loc[ps_group.index, "exclusion_reason"] = (
                "plantseg_internal_label_conflict"
            )
        else:
            kept_ps = preferred_index(ps_group)
            records.loc[kept_ps, "keep"] = True
            duplicate_ps = ps_group.index.difference([kept_ps])
            records.loc[duplicate_ps, "exclusion_reason"] = (
                "plantseg_duplicate_same_class"
            )

    if not pd_group.empty:
        if not ps_group.empty:
            ps_classes = set(ps_group["harmonized_class"])
            pd_classes = set(pd_group["harmonized_class"])
            reason = (
                "cross_dataset_label_conflict"
                if ps_classes != pd_classes
                else "plantdoc_overlaps_plantseg"
            )
            records.loc[pd_group.index, "exclusion_reason"] = reason
        else:
            pd_classes = set(pd_group["harmonized_class"])
            if len(pd_classes) > 1:
                records.loc[pd_group.index, "exclusion_reason"] = (
                    "plantdoc_internal_label_conflict"
                )
            else:
                kept_pd = preferred_index(pd_group)
                records.loc[kept_pd, "keep"] = True
                duplicate_pd = pd_group.index.difference([kept_pd])
                records.loc[duplicate_pd, "exclusion_reason"] = (
                    "plantdoc_duplicate_same_class"
                )

singleton_clusters = records.groupby("duplicate_cluster").size()
singleton_ids = set(singleton_clusters[singleton_clusters == 1].index)
singleton_mask = records["duplicate_cluster"].isin(singleton_ids)
records.loc[singleton_mask, "keep"] = True

assert (~records["keep"] | (records["exclusion_reason"] == "")).all()
assert (records["keep"] | (records["exclusion_reason"] != "")).all()

exclusions = records[~records["keep"]].copy()
display(
    exclusions.groupby(
        ["dataset", "exclusion_reason"], dropna=False
    ).size().reset_index(name="excluded_images")
)
exclusions.to_csv(OUTPUT_ROOT / "exclusions.csv", index=False)
records.to_csv(OUTPUT_ROOT / "all_records_with_decisions.csv", index=False)


clean_ps_training = records[
    records["keep"]
    & (records["dataset"] == "PlantSeg")
    & (records["source_split"] == "training")
].copy()

clean_ps_test = records[
    records["keep"]
    & (records["dataset"] == "PlantSeg")
    & (records["source_split"] == "test")
].copy()

train_positions, validation_positions = train_test_split(
    np.arange(len(clean_ps_training)),
    test_size=VALIDATION_FRACTION,
    random_state=SEED,
    stratify=clean_ps_training["harmonized_class"],
)

optimization_train = clean_ps_training.iloc[train_positions].copy()
optimization_validation = clean_ps_training.iloc[validation_positions].copy()
plantdoc_external_all = records[
    records["keep"] & (records["dataset"] == "PlantDoc")
].copy()

manifests = {
    "plantseg_optimization_train.csv": optimization_train,
    "plantseg_optimization_validation.csv": optimization_validation,
    "plantseg_official_test.csv": clean_ps_test,
    "plantdoc_external_all.csv": plantdoc_external_all,
}

for filename, frame in manifests.items():
    frame.to_csv(OUTPUT_ROOT / filename, index=False)

manifest_counts = []
for filename, frame in manifests.items():
    for class_name, count in frame.groupby("harmonized_class").size().items():
        manifest_counts.append({
            "manifest": filename,
            "harmonized_class": class_name,
            "images": int(count),
        })

manifest_count_table = pd.DataFrame(manifest_counts)
display(manifest_count_table.pivot(
    index="harmonized_class", columns="manifest", values="images"
).fillna(0).astype(int))
manifest_count_table.to_csv(
    OUTPUT_ROOT / "final_manifest_class_counts.csv", index=False
)

for filename, frame in manifests.items():
    assert set(frame["harmonized_class"]) == set(EXPECTED_CLASSES), (
        filename, sorted(set(EXPECTED_CLASSES) - set(frame["harmonized_class"]))
    )

print("\nManifest totals:")
for filename, frame in manifests.items():
    print(f"{filename}: {len(frame)}")


role_frames = {
    "train": optimization_train,
    "validation": optimization_validation,
    "plantseg_test": clean_ps_test,
    "plantdoc_external": plantdoc_external_all,
}

cluster_roles = defaultdict(set)
for role, frame in role_frames.items():
    for cluster_id in frame["duplicate_cluster"]:
        cluster_roles[int(cluster_id)].add(role)

crossing = {
    cluster_id: sorted(roles)
    for cluster_id, roles in cluster_roles.items()
    if len(roles) > 1
}

assert not crossing, f"Duplicate clusters cross final roles: {crossing}"
print("PASS: no detected duplicate cluster crosses final data roles.")

assert set(optimization_train["record_id"]).isdisjoint(
    optimization_validation["record_id"]
)
assert set(clean_ps_test["record_id"]).isdisjoint(
    optimization_train["record_id"]
)
assert set(clean_ps_test["record_id"]).isdisjoint(
    optimization_validation["record_id"]
)
print("PASS: final manifests have disjoint record identifiers.")


record_lookup = records.set_index("record_id")


def load_record_image(record, archive):
    if record.storage == "zip":
        data = archive.read(record.image_reference)
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(record.image_reference).convert("RGB")


review_pairs = pairs[
    (~pairs["same_harmonized_class"])
    | (pairs["relation"] == "PlantSeg vs PlantDoc")
    | (
        (pairs["relation"] == "PlantSeg internal")
        & (pairs["left_source_split"] != pairs["right_source_split"])
    )
].copy().reset_index(drop=True)

page_size = 10
with zipfile.ZipFile(ORIGINAL_PS_ZIP) as archive:
    for page_start in range(0, len(review_pairs), page_size):
        page = review_pairs.iloc[page_start:page_start + page_size]
        figure, axes = plt.subplots(
            len(page), 2, figsize=(9, 3.1 * len(page)), squeeze=False
        )
        for row_number, pair in enumerate(page.itertuples(index=False)):
            left = record_lookup.loc[pair.left_record_id]
            right = record_lookup.loc[pair.right_record_id]
            with load_record_image(left, archive) as image:
                axes[row_number, 0].imshow(image)
            with load_record_image(right, archive) as image:
                axes[row_number, 1].imshow(image)
            axes[row_number, 0].set_title(
                f"{left.dataset}/{left.source_split}\n{left.harmonized_class}",
                fontsize=8,
            )
            axes[row_number, 1].set_title(
                f"{right.dataset}/{right.source_split}\n"
                f"{right.harmonized_class}\nd={pair.phash_distance}",
                fontsize=8,
            )
            axes[row_number, 0].axis("off")
            axes[row_number, 1].axis("off")
        page_number = page_start // page_size + 1
        figure.suptitle(
            f"Duplicate/conflict review — page {page_number}", fontsize=14
        )
        figure.tight_layout()
        output_path = REVIEW_ROOT / f"review_page_{page_number:02d}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print("Saved:", output_path.name)

print("Review pairs:", len(review_pairs))
print("Review pages:", (len(review_pairs) + page_size - 1) // page_size)


pair_summary = (
    pairs.groupby(
        ["relation", "byte_identical", "same_harmonized_class"],
        dropna=False,
    ).size().reset_index(name="pairs").to_dict("records")
)
exclusion_summary = (
    exclusions.groupby(["dataset", "exclusion_reason"])
    .size().reset_index(name="images").to_dict("records")
)

protocol_summary = {
    "version": "clean_protocol_v1",
    "source_of_truth": {
        "plantseg": str(ORIGINAL_PS_ZIP),
        "plantdoc": str(ORIGINAL_PD_ROOT),
    },
    "phash_threshold": PHASH_THRESHOLD,
    "split_seed": SEED,
    "validation_fraction_of_clean_official_training": VALIDATION_FRACTION,
    "original_selected_counts": {
        "plantseg_training": int((
            (records["dataset"] == "PlantSeg")
            & (records["source_split"] == "training")
        ).sum()),
        "plantseg_test": int((
            (records["dataset"] == "PlantSeg")
            & (records["source_split"] == "test")
        ).sum()),
        "plantdoc_all": int((records["dataset"] == "PlantDoc").sum()),
    },
    "final_manifest_counts": {
        filename: int(len(frame)) for filename, frame in manifests.items()
    },
    "candidate_pair_summary": pair_summary,
    "exclusion_summary": exclusion_summary,
    "review_pairs": int(len(review_pairs)),
    "review_pages": int((len(review_pairs) + page_size - 1) // page_size),
    "sources_modified": False,
    "status": "PROVISIONAL_PENDING_VISUAL_REVIEW",
}

with open(OUTPUT_ROOT / "protocol_summary.json", "w", encoding="utf-8") as file:
    json.dump(protocol_summary, file, indent=2)

print(json.dumps(protocol_summary, indent=2))
