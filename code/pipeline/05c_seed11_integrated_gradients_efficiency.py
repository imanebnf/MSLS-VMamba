# Integrated Gradients and latency and memory measurements of the seed-11 models; fills runs/seed11_ig_efficiency.
import os

STORAGE = os.environ.get("MSLS_STORAGE", "storage")
SCRATCH = os.environ.get("MSLS_SCRATCH", "scratch")


def display(obj):
    print(obj)


import json
import os
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from captum.attr import IntegratedGradients
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageClassification
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

CONFIG = {
    'seed': 11,
    'img_size': 384,
    'batch_size': 2,
    'num_workers': 2,
    'ig_steps': 32,
    'deletion_fractions': (0.10, 0.20),
    'manifest': STORAGE + '/msls_vmamba/data/clean_protocol_v1/plantseg_optimization_validation.csv',
    'class_mapping': STORAGE + '/msls_vmamba/data/class_mapping.json',
    'plantseg_zip': STORAGE + '/plantseg.zip',
    'baseline_checkpoint': STORAGE + '/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1/cleanv1_vmamba_baseline_strong_384_seed11/best_model.pth',
    'current_checkpoint': STORAGE + '/msls_vmamba/runs/clean_protocol_v1_384/vmamba_msls_11_55_v1/cleanv1_current_full_strong_384_seed11/best_model.pth',
    'layercam_csv': STORAGE + '/msls_vmamba/runs/seed11_analyses/paired_layercam_seed11_v1/paired_layercam_per_image.csv',
    'layercam_selected_csv': STORAGE + '/msls_vmamba/runs/seed11_analyses/paired_layercam_seed11_v1/selected_layercam_examples.csv',
    'model_repo': 'saurabhati/VMamba_ImageNet_82.6',
    'model_dir': SCRATCH + '/vmamba_model',
    'cache_root': SCRATCH + '/vmamba_ig_validation',
    'output_root': STORAGE + '/msls_vmamba/runs/seed11_analyses/integrated_gradients_seed11_v1',
}

assert torch.cuda.is_available(), 'Select a GPU runtime.'
DEVICE = 'cuda'
for key in ('manifest', 'class_mapping', 'plantseg_zip', 'baseline_checkpoint', 'current_checkpoint'):
    assert Path(CONFIG[key]).exists(), f'Missing {key}: {CONFIG[key]}'
assert 'validation' in CONFIG['manifest'].lower()
assert 'official_test' not in CONFIG['manifest'].lower()
assert 'plantdoc' not in CONFIG['manifest'].lower()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OUTPUT = Path(CONFIG['output_root'])
OUTPUT.mkdir(parents=True, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0))
print('Guard: optimization-validation only; no official test or PlantDoc.')


with open(CONFIG['class_mapping'], encoding='utf-8') as handle:
    mapping = json.load(handle)
CLASS_NAMES = list(mapping['class_names'])
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
assert NUM_CLASSES == 16

frame = pd.read_csv(CONFIG['manifest'])
assert len(frame) == 350 and frame.record_id.is_unique
frame['label'] = frame.harmonized_class.map(CLASS_TO_INDEX).astype(int)

cache = Path(CONFIG['cache_root'])
cache.mkdir(parents=True, exist_ok=True)
local_images = []
local_masks = []
with zipfile.ZipFile(CONFIG['plantseg_zip']) as archive:
    members = set(archive.namelist())
    for row in frame.itertuples(index=False):
        assert row.image_reference in members, row.image_reference
        assert row.mask_reference in members, row.mask_reference
        image_path = cache / row.image_reference
        mask_path = cache / row.mask_reference
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            with archive.open(row.image_reference) as source, image_path.open('wb') as target:
                shutil.copyfileobj(source, target)
        if not mask_path.exists():
            with archive.open(row.mask_reference) as source, mask_path.open('wb') as target:
                shutil.copyfileobj(source, target)
        local_images.append(str(image_path))
        local_masks.append(str(mask_path))

frame['local_image'] = local_images
frame['local_mask'] = local_masks
print('Validation pairs cached:', len(frame))


def resize_and_pad(image, size, interpolation, fill):
    width, height = image.size
    scale = float(size) / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    image = TF.resize(
        image, [new_height, new_width], interpolation=interpolation,
        antialias=(interpolation != InterpolationMode.NEAREST),
    )
    left = (size - new_width) // 2
    right = size - new_width - left
    top = (size - new_height) // 2
    bottom = size - new_height - top
    return TF.pad(image, [left, top, right, bottom], fill=fill)


class ValidationSet(Dataset):
    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        with Image.open(row.local_image) as source:
            image = source.convert('RGB')
        with Image.open(row.local_mask) as source:
            mask = source.convert('L')
        image = resize_and_pad(
            image, CONFIG['img_size'], InterpolationMode.BICUBIC, (124, 116, 104)
        )
        mask = resize_and_pad(
            mask, CONFIG['img_size'], InterpolationMode.NEAREST, 0
        )
        shown = np.asarray(image).copy()
        image = TF.normalize(
            TF.to_tensor(image), [.485, .456, .406], [.229, .224, .225]
        )
        mask = (TF.pil_to_tensor(mask).float() > 0).float()
        return image, mask, int(row.label), row.record_id, shown


loader = DataLoader(
    ValidationSet(frame), batch_size=CONFIG['batch_size'], shuffle=False,
    num_workers=CONFIG['num_workers'], pin_memory=True,
)
print('Batches:', len(loader))


if not os.path.exists(os.path.join(CONFIG['model_dir'], 'config.json')):
    snapshot_download(
        repo_id=CONFIG['model_repo'],
        local_dir=CONFIG['model_dir'],
        revision='3db342a1b00a91c350dca4df93a1645ad2ac9493',
    )


def pretrained_vmamba():
    return AutoModelForImageClassification.from_pretrained(
        CONFIG['model_dir'],
        trust_remote_code=True,
        local_files_only=True,
    ).vmamba


class VMambaBaseline(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()
        self.vmamba = pretrained_vmamba()
        self.classifier = nn.Linear(self.vmamba.num_features, number_of_classes)

    def forward(self, images):
        features = self.vmamba.patch_embeddings(images)
        for layer in self.vmamba.layers:
            features = layer(features)
        features = self.vmamba.norm(features)
        vector = self.vmamba.avgpool(features).flatten(1)
        return self.classifier(vector)


class NativeMeanFusionHead(nn.Module):
    def __init__(self, channels=(192, 384, 768), hidden=128):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, hidden, 1, bias=False),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
            for channel in channels
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

    def forward(self, features):
        vectors = []
        for projection, predictor, feature in zip(
            self.projections, self.mask_predictors, features
        ):
            projected = projection(feature)
            weight = torch.sigmoid(predictor(projected))
            vectors.append(
                (projected * weight).sum((2, 3))
                / weight.sum((2, 3)).clamp_min(1e-6)
            )
        return vectors


class CurrentFull(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()
        self.vmamba = pretrained_vmamba()
        dimension = self.vmamba.num_features
        self.region_head = NativeMeanFusionHead()
        self.lesion_projection = nn.Sequential(
            nn.Linear(384, dimension),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.fusion_norm = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, number_of_classes)

    def forward(self, images):
        features = self.vmamba.patch_embeddings(images)
        multiscale = []
        for index in range(3):
            features = self.vmamba.layers[index](features)
            multiscale.append(features)
        lesion = self.lesion_projection(
            torch.cat(self.region_head(multiscale), dim=1)
        )
        features = self.vmamba.layers[3](features)
        features = self.vmamba.norm(features)
        global_vector = self.vmamba.avgpool(features).flatten(1)
        return self.classifier(self.fusion_norm(global_vector + lesion))


def load_checkpoint(model, path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict):
        for key in ('model_state_dict', 'model_state', 'state_dict'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    return model.to(device).eval()

baseline_cpu = load_checkpoint(
    VMambaBaseline(NUM_CLASSES), CONFIG['baseline_checkpoint'], device='cpu'
)
current_cpu = load_checkpoint(
    CurrentFull(NUM_CLASSES), CONFIG['current_checkpoint'], device='cpu'
)
print('Both checkpoints loaded on CPU.')


def profile_model(name, model, checkpoint_path, batch_sizes=(1, 8)):
    model = model.to(DEVICE).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    output_rows = []
    for batch_size in batch_sizes:
        sample = torch.randn(
            batch_size, 3, CONFIG['img_size'], CONFIG['img_size'], device=DEVICE
        )
        with torch.inference_mode():
            for _ in range(15):
                model(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        repetitions = 60
        start.record()
        with torch.inference_mode():
            for _ in range(repetitions):
                model(sample)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / repetitions
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        output_rows.append({
            'model': name,
            'batch_size': batch_size,
            'parameters_M': parameter_count / 1e6,
            'trainable_parameters_M': trainable_count / 1e6,
            'checkpoint_MB': Path(checkpoint_path).stat().st_size / 1024 ** 2,
            'batch_latency_ms': elapsed_ms,
            'latency_ms_per_image': elapsed_ms / batch_size,
            'throughput_images_per_second': 1000.0 * batch_size / elapsed_ms,
            'peak_gpu_memory_MB': peak_mb,
        })
        del sample
    model = model.to('cpu')
    torch.cuda.empty_cache()
    return output_rows


efficiency_rows = []
efficiency_rows.extend(profile_model(
    'vmamba_baseline', baseline_cpu, CONFIG['baseline_checkpoint']
))
efficiency_rows.extend(profile_model(
    'current_full', current_cpu, CONFIG['current_checkpoint']
))
efficiency = pd.DataFrame(efficiency_rows)
display(efficiency.round(3))
efficiency.to_csv(OUTPUT / 'measured_efficiency_seed11.csv', index=False)

baseline = baseline_cpu.to(DEVICE).eval()
current = current_cpu.to(DEVICE).eval()


baseline_ig = IntegratedGradients(baseline)
current_ig = IntegratedGradients(current)


def positive_map(attributions):
    return F.relu(attributions.sum(dim=1, keepdim=True))


def attribution_metrics(attribution, masks):
    epsilon = 1e-8
    binary = masks > 0.5
    total = attribution.sum((1, 2, 3)).clamp_min(epsilon)
    inside = (attribution * binary).sum((1, 2, 3))
    energy = inside / total
    lesion_mean = inside / binary.sum((1, 2, 3)).clamp_min(1)
    global_mean = attribution.mean((1, 2, 3)).clamp_min(epsilon)
    concentration = lesion_mean / global_mean
    flat_index = attribution.flatten(1).argmax(1)
    pointing = binary.flatten(1).gather(1, flat_index[:, None]).squeeze(1).float()
    lesion_area = binary.float().mean((1, 2, 3))
    return energy, concentration, pointing, lesion_area


def integrated_gradients_for_batch(model, explainer, images):
    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(1)
        original_probabilities = torch.softmax(logits, dim=1).gather(
            1, predictions[:, None]
        ).squeeze(1)
    attributions, convergence_delta = explainer.attribute(
        images,
        baselines=torch.zeros_like(images),
        target=predictions,
        n_steps=CONFIG['ig_steps'],
        method='gausslegendre',
        internal_batch_size=max(len(images), 8),
        return_convergence_delta=True,
    )
    return (
        logits.detach(), predictions.detach(),
        positive_map(attributions).detach(),
        convergence_delta.detach(), original_probabilities.detach(),
    )


def delete_top_pixels(model, images, attribution, target, original_probability, fraction):
    batch, _, height, width = images.shape
    flat = attribution.flatten(1)
    number = max(1, int(round(fraction * height * width)))
    top = flat.topk(number, dim=1).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, top, True)
    mask = mask.view(batch, 1, height, width)
    blurred = TF.gaussian_blur(images, kernel_size=[31, 31], sigma=[10.0, 10.0])
    deleted = torch.where(mask, blurred, images)
    with torch.no_grad():
        probability = torch.softmax(model(deleted), dim=1).gather(
            1, target[:, None]
        ).squeeze(1)
    return (original_probability - probability).detach()


images, masks, labels, record_ids, shown = next(iter(loader))
images = images[:1].to(DEVICE)
_, _, smoke_attr, smoke_delta, smoke_probability = integrated_gradients_for_batch(
    baseline, baseline_ig, images
)
assert smoke_attr.shape == (1, 1, 384, 384)
assert torch.isfinite(smoke_attr).all()
print('Integrated Gradients smoke test: PASS', smoke_attr.shape)
del images, smoke_attr, smoke_delta, smoke_probability
torch.cuda.empty_cache()


progress_path = OUTPUT / 'ig_per_image_progress.csv'
if progress_path.exists():
    progress = pd.read_csv(progress_path)
    completed_ids = set(progress.record_id)
    rows = progress.to_dict('records')
    print('Resuming completed images:', len(completed_ids))
else:
    completed_ids = set()
    rows = []

for batch_index, (images, masks, labels, record_ids, shown) in enumerate(loader):
    keep = [index for index, record_id in enumerate(record_ids) if record_id not in completed_ids]
    if not keep:
        continue
    images = images[keep].to(DEVICE, non_blocking=True)
    masks = masks[keep].to(DEVICE, non_blocking=True)
    labels = labels[keep]
    ids = [record_ids[index] for index in keep]

    b_logits, b_prediction, b_attr, b_delta, b_probability = integrated_gradients_for_batch(
        baseline, baseline_ig, images
    )
    c_logits, c_prediction, c_attr, c_delta, c_probability = integrated_gradients_for_batch(
        current, current_ig, images
    )

    b_energy, b_concentration, b_pointing, lesion_area = attribution_metrics(b_attr, masks)
    c_energy, c_concentration, c_pointing, _ = attribution_metrics(c_attr, masks)

    deletion = {}
    for fraction in CONFIG['deletion_fractions']:
        key = int(round(100 * fraction))
        deletion[f'b_{key}'] = delete_top_pixels(
            baseline, images, b_attr, b_prediction, b_probability, fraction
        )
        deletion[f'c_{key}'] = delete_top_pixels(
            current, images, c_attr, c_prediction, c_probability, fraction
        )

    for local_index, record_id in enumerate(ids):
        true_index = int(labels[local_index])
        baseline_index = int(b_prediction[local_index])
        current_index = int(c_prediction[local_index])
        row = {
            'record_id': record_id,
            'true_class': CLASS_NAMES[true_index],
            'baseline_prediction': CLASS_NAMES[baseline_index],
            'current_prediction': CLASS_NAMES[current_index],
            'baseline_correct': baseline_index == true_index,
            'current_correct': current_index == true_index,
            'lesion_area_ratio': float(lesion_area[local_index]),
            'baseline_ig_energy': float(b_energy[local_index]),
            'current_ig_energy': float(c_energy[local_index]),
            'delta_ig_energy': float(c_energy[local_index] - b_energy[local_index]),
            'baseline_ig_concentration': float(b_concentration[local_index]),
            'current_ig_concentration': float(c_concentration[local_index]),
            'delta_ig_concentration': float(c_concentration[local_index] - b_concentration[local_index]),
            'baseline_ig_pointing': float(b_pointing[local_index]),
            'current_ig_pointing': float(c_pointing[local_index]),
            'baseline_completeness_abs_error': float(abs(b_delta[local_index])),
            'current_completeness_abs_error': float(abs(c_delta[local_index])),
        }
        for fraction in CONFIG['deletion_fractions']:
            key = int(round(100 * fraction))
            row[f'baseline_top{key}_deletion_probability_drop'] = float(deletion[f'b_{key}'][local_index])
            row[f'current_top{key}_deletion_probability_drop'] = float(deletion[f'c_{key}'][local_index])
        rows.append(row)
        completed_ids.add(record_id)

    pd.DataFrame(rows).drop_duplicates('record_id', keep='last').to_csv(
        progress_path, index=False
    )
    if (batch_index + 1) % 10 == 0:
        print('Completed:', len(completed_ids), '/350')

results = pd.DataFrame(rows).drop_duplicates('record_id', keep='last')
assert len(results) == 350, len(results)
results['transition'] = np.select(
    [
        results.baseline_correct & results.current_correct,
        (~results.baseline_correct) & results.current_correct,
        results.baseline_correct & (~results.current_correct),
    ],
    ['both_correct', 'current_fixes_baseline', 'current_breaks_baseline'],
    default='both_wrong',
)
results.to_csv(OUTPUT / 'integrated_gradients_per_image.csv', index=False)
print('Complete paired attribution table saved.')


subsets = {
    'all': results,
    'both_correct': results[results.transition == 'both_correct'],
    'current_fixes_baseline': results[results.transition == 'current_fixes_baseline'],
    'current_breaks_baseline': results[results.transition == 'current_breaks_baseline'],
}
summary_rows = []
for subset_name, subset in subsets.items():
    if subset.empty:
        continue
    summary_rows.append({
        'subset': subset_name,
        'images': len(subset),
        'baseline_concentration_mean': subset.baseline_ig_concentration.mean(),
        'current_concentration_mean': subset.current_ig_concentration.mean(),
        'mean_delta_concentration': subset.delta_ig_concentration.mean(),
        'percent_current_more_concentrated': 100 * (subset.delta_ig_concentration > 0).mean(),
        'baseline_top10_deletion_drop_mean': subset.baseline_top10_deletion_probability_drop.mean(),
        'current_top10_deletion_drop_mean': subset.current_top10_deletion_probability_drop.mean(),
        'mean_delta_top10_deletion_drop': (
            subset.current_top10_deletion_probability_drop
            - subset.baseline_top10_deletion_probability_drop
        ).mean(),
    })

summary = pd.DataFrame(summary_rows)
display(summary.round(4))
summary.to_csv(OUTPUT / 'integrated_gradients_summary.csv', index=False)

layercam_path = Path(CONFIG['layercam_csv'])
agreement = {}
if layercam_path.exists():
    layercam = pd.read_csv(layercam_path)[['record_id', 'delta_concentration']]
    merged = results.merge(layercam, on='record_id', how='inner')
    correlation, _ = spearmanr(
        merged.delta_ig_concentration, merged.delta_concentration
    )
    agreement = {
        'images': len(merged),
        'spearman_ig_vs_layercam_delta': float(correlation),
        'percent_same_positive_direction': float(100 * (
            (merged.delta_ig_concentration > 0)
            & (merged.delta_concentration > 0)
        ).mean()),
    }
    print('LayerCAM agreement:', agreement)


sns.set_theme(style='whitegrid', context='talk')
figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))

concentration_plot = pd.DataFrame({
    'VMamba': results.baseline_ig_concentration,
    'Current full': results.current_ig_concentration,
}).melt(var_name='Model', value_name='IG lesion concentration')
sns.violinplot(
    data=concentration_plot, x='Model', y='IG lesion concentration',
    inner='quartile', cut=0, ax=axes[0], palette=['#A5A5A5', '#4472C4']
)
axes[0].axhline(1, color='black', linewidth=1)
axes[0].set_title('Input attribution concentration')

deletion_plot = pd.DataFrame({
    'VMamba': 100 * results.baseline_top10_deletion_probability_drop,
    'Current full': 100 * results.current_top10_deletion_probability_drop,
}).melt(var_name='Model', value_name='Probability drop (pp)')
sns.boxplot(
    data=deletion_plot, x='Model', y='Probability drop (pp)',
    ax=axes[1], palette=['#A5A5A5', '#4472C4']
)
axes[1].set_title('Delete top 10% attributed pixels')

extent = pd.cut(
    results.lesion_area_ratio,
    bins=[0, .05, .15, .30, 1.0],
    labels=['tiny 0–5%', 'small 5–15%', 'medium 15–30%', 'large 30–100%'],
    include_lowest=True,
)
extent_delta = results.assign(extent=extent).groupby(
    'extent', observed=False
).delta_ig_concentration.mean()
sns.barplot(x=extent_delta.index, y=extent_delta.values, ax=axes[2], color='#70AD47')
axes[2].set_title('IG change by lesion extent')
axes[2].set_xlabel('Ground-truth lesion extent')
axes[2].set_ylabel('Current − baseline concentration')
axes[2].tick_params(axis='x', rotation=25)

figure.tight_layout()
figure.savefig(OUTPUT / 'integrated_gradients_aggregate.png', dpi=300, bbox_inches='tight')
plt.show()


selection_parts = []
selection_rules = [
    ('largest current fixes', 'current_fixes_baseline', False, 2),
    ('largest both-correct shifts', 'both_correct', False, 2),
    ('smallest shifts (counterexamples)', None, True, 2),
]
for selection_name, transition_name, ascending, number in selection_rules:
    candidates = results if transition_name is None else results[
        results.transition == transition_name
    ]
    selected = candidates.sort_values(
        ['delta_ig_concentration', 'record_id'],
        ascending=[ascending, True],
    ).head(number).copy()
    selected['selection_rule'] = selection_name
    selection_parts.append(selected)

selected_examples = pd.concat(selection_parts, ignore_index=True)
selected_examples = selected_examples.drop_duplicates('record_id', keep='first')
if len(selected_examples) < 6:
    supplements = results.loc[
        ~results.record_id.isin(selected_examples.record_id)
    ].sort_values(
        ['delta_ig_concentration', 'record_id'], ascending=[False, True]
    ).head(6 - len(selected_examples)).copy()
    supplements['selection_rule'] = 'deterministic fill'
    selected_examples = pd.concat(
        [selected_examples, supplements], ignore_index=True
    )
selected_examples = selected_examples.head(6).reset_index(drop=True)
selected_examples.to_csv(OUTPUT / 'selected_ig_examples.csv', index=False)


def normalized_heatmap(attribution):
    heatmap = attribution.squeeze().detach().cpu().numpy()
    upper = np.percentile(heatmap, 99)
    if upper <= 0:
        return np.zeros_like(heatmap)
    return np.clip(heatmap / upper, 0, 1)


validation_dataset = ValidationSet(frame)
record_to_index = {
    record_id: index for index, record_id in enumerate(frame.record_id)
}
figure, axes = plt.subplots(
    len(selected_examples), 4,
    figsize=(15, 3.7 * len(selected_examples)), squeeze=False,
)

for plot_index, selected_row in selected_examples.iterrows():
    dataset_index = record_to_index[selected_row.record_id]
    image, mask, label, record_id, shown = validation_dataset[dataset_index]
    image_batch = image.unsqueeze(0).to(DEVICE)
    mask_array = mask.squeeze().numpy().astype(bool)

    _, baseline_prediction, baseline_attr, _, _ = integrated_gradients_for_batch(
        baseline, baseline_ig, image_batch
    )
    _, current_prediction, current_attr, _, _ = integrated_gradients_for_batch(
        current, current_ig, image_batch
    )

    mask_overlay = shown.astype(np.float32).copy()
    red = np.zeros_like(mask_overlay)
    red[..., 0] = 255
    mask_overlay[mask_array] = (
        0.55 * mask_overlay[mask_array] + 0.45 * red[mask_array]
    )

    baseline_map = normalized_heatmap(baseline_attr)
    current_map = normalized_heatmap(current_attr)
    titles = [
        f"Original\ntrue: {CLASS_NAMES[label]}",
        f"Ground-truth lesion\narea: {100 * mask_array.mean():.1f}%",
        f"VMamba IG\npred: {CLASS_NAMES[int(baseline_prediction.item())]}",
        f"Current-full IG\npred: {CLASS_NAMES[int(current_prediction.item())]}",
    ]
    panels = [shown, mask_overlay.astype(np.uint8), shown, shown]
    for column, (panel, title) in enumerate(zip(panels, titles)):
        axes[plot_index, column].imshow(panel)
        if column == 2:
            axes[plot_index, column].imshow(
                baseline_map, cmap='jet', alpha=0.45, vmin=0, vmax=1
            )
        elif column == 3:
            axes[plot_index, column].imshow(
                current_map, cmap='jet', alpha=0.45, vmin=0, vmax=1
            )
        axes[plot_index, column].set_title(title, fontsize=9)
        axes[plot_index, column].axis('off')

    axes[plot_index, 0].set_ylabel(
        f"{selected_row.selection_rule}\n"
        f"delta={selected_row.delta_ig_concentration:+.2f}",
        fontsize=9,
    )
    del image_batch, baseline_attr, current_attr

figure.suptitle(
    'Integrated Gradients: deterministic examples and counterexamples',
    fontsize=15, y=1.002,
)
figure.tight_layout()
figure.savefig(
    OUTPUT / 'integrated_gradients_examples.png',
    dpi=300, bbox_inches='tight',
)
plt.show()
torch.cuda.empty_cache()
display(selected_examples[[
    'selection_rule', 'record_id', 'transition',
    'delta_ig_concentration',
]])


all_row = summary.loc[summary.subset == 'all'].iloc[0]
efficiency_wide = efficiency.pivot(
    index='batch_size', columns='model',
    values=['parameters_M', 'latency_ms_per_image',
            'throughput_images_per_second', 'peak_gpu_memory_MB'],
)
batch_one = efficiency_wide.loc[1]
measured_efficiency = {
    'parameter_overhead_percent': float(100 * (
        batch_one[('parameters_M', 'current_full')]
        / batch_one[('parameters_M', 'vmamba_baseline')] - 1
    )),
    'batch1_latency_overhead_percent': float(100 * (
        batch_one[('latency_ms_per_image', 'current_full')]
        / batch_one[('latency_ms_per_image', 'vmamba_baseline')] - 1
    )),
    'batch1_peak_memory_overhead_percent': float(100 * (
        batch_one[('peak_gpu_memory_MB', 'current_full')]
        / batch_one[('peak_gpu_memory_MB', 'vmamba_baseline')] - 1
    )),
}
supports_layercam = bool(
    all_row.mean_delta_concentration > 0
    and all_row.percent_current_more_concentrated > 50
)
report = {
    'version': 'integrated_gradients_seed11_v1',
    'seed': 11,
    'images': len(results),
    'ig_steps': CONFIG['ig_steps'],
    'mean_delta_ig_concentration': float(all_row.mean_delta_concentration),
    'percent_images_current_more_concentrated': float(all_row.percent_current_more_concentrated),
    'mean_delta_top10_deletion_probability_drop': float(all_row.mean_delta_top10_deletion_drop),
    'layercam_agreement': agreement,
    'measured_efficiency': measured_efficiency,
    'official_tests_evaluated': False,
    'plantdoc_evaluated': False,
}
with (OUTPUT / 'integrated_gradients_report.json').open('w') as handle:
    json.dump(report, handle, indent=2)
print('=' * 96)
print(json.dumps(report, indent=2))
print('Saved:', OUTPUT)
print('=' * 96)


import importlib, subprocess, sys, tempfile
try:
    import timm
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'timm==0.9.16', '--no-deps'])
    import timm
import torchvision.models as tvm


def state_dict_mb(model):
    with tempfile.NamedTemporaryFile(suffix='.pth') as handle:
        torch.save(model.state_dict(), handle.name)
        return Path(handle.name).stat().st_size / 1024 ** 2


def resnet50_16():
    model = tvm.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def effnetv2s_16():
    model = tvm.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def vits16_16():
    return timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES, img_size=CONFIG['img_size'])


all_rows = []
for name, model in (('vmamba_baseline', baseline.cpu()), ('current_full', current.cpu()),
                    ('resnet50', resnet50_16()), ('efficientnet_v2_s', effnetv2s_16()), ('vit_s16', vits16_16())):
    size_mb = state_dict_mb(model)
    with tempfile.NamedTemporaryFile(suffix='.pth') as handle:
        torch.save(model.state_dict(), handle.name)
        rows = profile_model(name, model, handle.name)
    for row in rows:
        row['state_dict_MB'] = size_mb
        row['gpu'] = torch.cuda.get_device_name(0)
    all_rows.extend(rows)
    model.cpu()
all_efficiency = pd.DataFrame(all_rows)
display(all_efficiency.round(3))
all_efficiency.to_csv(OUTPUT / 'measured_efficiency_all_models_seed11.csv', index=False)
baseline = baseline.to(DEVICE).eval(); current = current.to(DEVICE).eval()
print('Saved', OUTPUT / 'measured_efficiency_all_models_seed11.csv')
