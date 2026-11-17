# MSLS-VMamba

Code and data records for the article *Teaching a Plant Disease Classifier Where to Look: Multiscale Lesion-Supervised
VMamba for Field Images*.

Images are not redistributed: they are identified by their names in the original PlantSeg and PlantDoc archives.
Checkpoints are available on request because of their size.

## Recompute the tables and figures (CPU)

From the repository root (`OUT` is any output folder):

```bash
pip install -r requirements-analysis.txt
python code/check_files.py
python code/analysis/main_table.py OUT/main runs/vmamba_msls_seeds11-55 runs/baselines_seeds11-55
python code/analysis/crop_errors_five_seeds.py runs/vmamba_msls_seeds11-55 OUT/main
python code/analysis/msvmamba_comparison.py OUT/msvmamba_comparison runs/vmamba_msls_seeds11-55 runs/msvmamba_tiny_seeds11-55
python code/analysis/spatial_controls.py OUT/spatial_controls runs/vmamba_msls_seeds11-55 runs/spatial_controls_seeds11-55
python code/analysis/seed11_ablation.py runs/seed11_locked_eval_interventions/seed11_locked_eval_summary.csv OUT/seed11
python code/analysis/seed11_local.py runs/vmamba_msls_seeds11-55 data/plantseg_test_corrected_mask_references.csv OUT/seed11
python code/analysis/table7_attribution.py OUT/seed11
python code/analysis/supplementary_tables.py OUT/supplementary_tables
python code/analysis/efficiency_table.py runs/seed11_ig_efficiency/measured_efficiency_all_models_seed11.csv OUT/efficiency
python code/analysis/lesion_perturbation.py runs/lesion_perturbation_seeds11-55 runs/vmamba_msls_seeds11-55 OUT/lesion_perturbation
python code/analysis/alignment_interventions.py runs/alignment_interventions_test_seeds11-55 runs/vmamba_msls_seeds11-55 OUT/alignment_interventions
python code/analysis/layercam_by_extent_test.py runs/alignment_interventions_test_seeds11-55 runs/lesion_perturbation_seeds11-55/plantseg_test_regions.csv OUT/alignment_interventions
python code/figures/make_figures.py OUT/figures
```

Each command writes the tables of the folder of the same name in `analysis_outputs/`; `OUT/supplementary_tables`
corresponds to `supplementary_tables/`. The bootstrap intervals use a fixed seed, so the recorded tables are
reproduced.

## Run the pipeline (GPU)

1. Download PlantSeg (Wei et al., arXiv:2409.04038; archive `plantseg.zip`) and PlantDoc (Singh et al.,
   <https://doi.org/10.1145/3371158.3371196>; folder `PlantDoc-Dataset`).
2. Install torch 2.2.2 and torchvision 0.17.2 for CUDA 12.1 (<https://download.pytorch.org/whl/cu121>), then
   `pip install -r requirements-pipeline.txt`.
3. Set `MSLS_STORAGE` to a folder holding `plantseg.zip` and `PlantDoc-Dataset/`, and copy `data/class_mapping.json` to
   `$MSLS_STORAGE/msls_vmamba/data/`. Set `MSLS_SCRATCH` to a local cache folder.
4. Run the scripts in `code/pipeline/` in numerical order. The pretrained weights and the MSVMamba code are downloaded
   automatically. Scripts 05a–05c, 06c and 06d require the trained checkpoints.

`data/manifests/plantseg_official_test.csv` is kept exactly as it was when the models were trained. Its
`mask_reference` column repeats one path for all 435 test images, so it does not identify their annotations; the
annotation of each test image is given in `data/plantseg_test_corrected_mask_references.csv`, and the scripts derive
it from the image name rather than from that column. The runs in `runs/` were produced by running this code as
notebooks; the scripts in `code/pipeline/` have not been re-run in their current form.
