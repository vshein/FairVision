# FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling

The code and dataset for the paper entitled [**FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling**](https://arxiv.org/pdf/2310.02492). Note that, the modifier word “Harvard” only indicates that our dataset is from the Department of Ophthalmology of Harvard Medical School and does not imply an endorsement, sponsorship, or assumption of responsibility by either Harvard University or Harvard Medical School as a legal identity.

## Quick start

```bash
git clone https://github.com/vshein/FairVision.git
cd FairVision
pip install -r requirements.txt          # all dependencies, no sudo needed
./scripts/prepare_data.sh DR            # extracts the DR archive into ./data
./scripts/train_dr_vit.sh               # Step 1: baseline ViT-B on DR
```

Nothing else has to be configured: `prepare_data.sh` looks for the already
downloaded Hugging Face dataset in the cache (your own, or another user's that
you can read), extracts it into the repository's `data/` folder, and every
training script reads that folder by default. If the dataset has never been
downloaded on this machine, run
`hf download harvardairobotics/FairVision --repo-type dataset` first (see
[§2](#2-preparing-the-data)).

**Already have the data extracted?** Then skip `prepare_data.sh` entirely and
just tell the scripts where it is — one environment variable, no file editing:

```bash
DATASET_DIR=/path/to/FairVision ./scripts/train_dr_vit.sh
```

Never edit a path inside the scripts. `DATASET_DIR` (dataset) and `RESULT_DIR`
(outputs) are the only two knobs you should ever need — both have sensible
defaults — and there is a third one, `MODELHUB_DIR`, only for the optional
`--vit_weights mae/mocov3/...` checkpoints. See
[§1.4](#14-point-the-code-at-the-data) for all the supported data layouts.

## Dataset

The dataset Harvard-FairVision can be accessed via this [link](https://huggingface.co/datasets/harvardairobotics/FairGenMed). This dataset can only be used for non-commercial research purposes. At no time, the dataset shall be used for clinical decisions or patient care. The data use license is [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). If you have any questions, please email <harvardophai@gmail.com> and <harvardairobotics@gmail.com>.

Our dataset includes 10,000 subjects for Age-Related Macular Degeneration (AMD), Diabetic Retinopathy (DR), and glaucoma separately, totaling 30,000 subjects with comprehensive demographic identity attributes including age, gender, race, ethnicity, preferred language, and marital status. Each subject has one Scanning Laser Ophthalmoscopy (SLO) fundus photo and one sample of Optical Coherence Tomography (OCT) B-scans. The size of OCT B-scans is 200 x 200 x 200 in glaucoma, while the one of OCT B-scans. The size of OCT B-scans is 128 x 200 x 200 in AMD and DR. 

The dataset has an approximate size of 600 GB. Upon downloading and extracting these datasets, the names of the folders in the downloaded dataset under folders AMD/DR/Glaucoma are Training, Validation, and Test, respectively, for readability. You will find the dataset structure as follows. 

```
FairVision
├── AMD
│   ├── Training
│   ├── Validation
│   └── Test
├── data_summary_amd.csv
├── DR
│   ├── Training
│   ├── Validation
│   └── Test
├── data_summary_dr.csv
├── Glaucoma
│   ├── Training
│   ├── Validation
│   └── Test
└── data_summary_glaucoma.csv
```
The "Training/Validation/Test" directories contain two types of data: SLO fundus photos and NPZ files that store OCT B-scans, SLO fundus photos, and additional attributes. SLO fundus photos serve visual inspection purposes, while the copies in NPZ files eliminate the need for the dataloader to access any other files except the NPZ files. The naming convention for SLO fundus photos follows the format "slo_xxxxx.jpg," and for NPZ files, it is "data_xxxxx.npz," where "xxxxx" (e.g., 07777) represents a unique numeric ID. The dimensions of SLO fundus photos in NPZ files are 200 x 200, whereas those in the train/val/test folders are 512 x 664. The SLO fundus photos in NPZ files are created by resizing the photos in the folders and then normalizing them to [0, 255].

NPZ files have the following keys. 

In the AMD disease, the NPZ files have
```
amd_condition: AMD conditions - {'not.in.icd.table', 'no.amd.diagnosis', 'early.dry', 'intermediate.dry', 'advanced.atrophic.dry.with.subfoveal.involvement', 'advanced.atrophic.dry.without.subfoveal.involvement', 'wet.amd.active.choroidal.neovascularization', 'wet.amd.inactive.choroidal.neovascularization', 'wet.amd.inactive.scar'}
oct_bscans: images of OCT B-scans
slo_fundus: image of SLO fundus
race: 0 - Asian, 1 - Black, 2 - White
male: 0 - Female, 1 - Male
hispanic: 0 - Non-Hispanic, 1 - Hispanic
maritalstatus: 0 - Married, 1 - Single, 2 - Divorced, 3 - Widowed, 4 - Leg-Sep
language: 0 - English, 1 - Spanish, 2 - Others
```
The condition would be converted into the label of AMD by the condition-disease mapping.
```
condition_disease_mapping = {'not.in.icd.table': 0.,
                        'no.amd.diagnosis': 0.,
                        'early.dry': 1.,
                        'intermediate.dry': 2.,
                        'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
                        'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
                        'wet.amd.active.choroidal.neovascularization': 3.,
                        'wet.amd.inactive.choroidal.neovascularization': 3.,
                        'wet.amd.inactive.scar': 3.}
```

In the DR disease, the NPZ files have
```
dr_subtype: DR conditions - {'not.in.icd.table', 'no.dr.diagnosis', 'mild.npdr', 'moderate.npdr', 'severe.npdr', 'pdr'}
oct_bscans: images of OCT B-scans
slo_fundus: image of SLO fundus
race: 0 - Asian, 1 - Black, 2 - White
male: 0 - Female, 1 - Male
hispanic: 0 - Non-Hispanic, 1 - Hispanic
maritalstatus: 0 - Married, 1 - Single, 2 - Divorced, 3 - Widowed, 4 - Leg-Sep
language: 0 - English, 1 - Spanish, 2 - Others
```
The condition would be converted into the label of vision-threatening DR by the condition-disease mapping.
```
condition_disease_mapping = {'not.in.icd.table': 0.,
                    'no.dr.diagnosis': 0.,
                    'mild.npdr': 0.,
                    'moderate.npdr': 0.,
                    'severe.npdr': 1.,
                    'pdr': 1.}
```

In the glaucoma disease, the NPZ files have
```
glaucoma: the label of glaucoma disease, 0 - non-glaucoma, 1 - glaucoma
oct_bscans: images of OCT B-scans
slo_fundus: image of SLO fundus
race: 0 - Asian, 1 - Black, 2 - White
male: 0 - Female, 1 - Male
hispanic: 0 - Non-Hispanic, 1 - Hispanic
maritalstatus: 0 - Married, 1 - Single, 2 - Divorced, 3 - Widowed, 4 - Leg-Sep
language: 0 - English, 1 - Spanish, 2 - Others
```

We put all the attributes associated with the 10,000 samples in a meta csv file for each disease, including race, gender, ethnicity, marital status, age, preferred language.


## Abstract

<p align="center">
<img src="fig/overview.png" width="500">
</p>

Equity in AI for healthcare is crucial due to its direct impact on human well-being. Despite advancements in 2D medical imaging fairness, the fairness of 3D models remains underexplored, hindered by the small sizes of 3D fairness datasets. Since 3D imaging surpasses 2D imaging in SOTA clinical care, it is critical to understand the fairness of these 3D models. To address this research gap, we conduct the first comprehensive study on the fairness of 3D medical imaging models across multiple protected attributes. Our investigation spans both 2D and 3D models and evaluates fairness across five architectures on three common eye diseases, revealing significant biases across race, gender, and ethnicity. To alleviate these biases, we propose a novel fair identity scaling (FIS) method that improves both overall performance and fairness, outperforming various SOTA fairness methods. Moreover, we release Harvard-FairVision, the first large-scale medical fairness dataset with 30,000 subjects featuring both 2D and 3D imaging data and six demographic identity attributes. Harvard-FairVision provides labels for three major eye disorders affecting about 380 million people worldwide, serving as a valuable resource for both 2D and 3D fairness learning.

## Table of contents

0. [Quick start](#quick-start)
1. [Environment setup](#1-environment-setup)
2. [Preparing the data](#2-preparing-the-data)
3. [Experiments (step by step)](#3-experiments)
4. [Troubleshooting](#4-troubleshooting)
5. [Acknowledgment and Citation](#acknowledgment-and-citation)

---

## 1. Environment setup

### 1.1 What you need

* Linux (the experiment scripts are `bash`); macOS works as well.
* Python 3.10+ — this checkout was validated with Python 3.13.
* Free disk space: ~55 GB per disease for the extracted data (~40 GB of archives
  in the Hugging Face cache) plus room for checkpoints and logs.
* A CUDA GPU is strongly recommended (the experiments train ViT-B and 3D ResNet
  models). The code also runs on CPU — `torch.cuda.is_available()` simply returns
  `False` and everything is executed on the CPU, only much slower.

> **No `sudo` is required for any step.** All Python packages are installed into
> your own environment and all extracted data, checkpoints and logs are written
> to folders you own.

### 1.2 Install the dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

`requirements.txt` lists every third-party package the repository imports:
`numpy`, `scipy`, `pandas`, `Pillow`, `blobfile`, `scikit-image`,
`opencv-python`, `einops`, `torch`, `torchvision`, `timm`, `transformers`,
`scikit-learn`, `fairlearn` and `geomloss`.

If you prefer an isolated environment:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**GPU / CPU note.** A plain `pip install torch` downloads the CUDA build (several
GB of `nvidia-*` wheels). On a machine without a GPU you can avoid that with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 1.3 Verify the installation

```bash
python -c "import torch, torchvision, timm, transformers, blobfile, einops, cv2, skimage, sklearn, fairlearn, geomloss; print('all dependencies OK')"
```

The experiments themselves are started with the same interpreter that printed
`all dependencies OK`.

### 1.4 Point the code at the data

Every script in `scripts/` reads the dataset root from the `DATASET_DIR`
environment variable. If it is **not** set (the normal case), the scripts use

```text
<repository>/data/
```

i.e. the folder that `scripts/prepare_data.sh` fills in. Because the scripts
resolve this relative to their own location, a fresh clone runs with **no path
edits at all** — you never have to edit a script or a Python file.

There are three situations; pick the one that matches your machine.

**(a) You are starting from scratch (only the Hugging Face cache exists).**
Do nothing — follow [Quick start](#quick-start) / [§2](#2-preparing-the-data):

```bash
./scripts/prepare_data.sh DR     # extracts the cache into <repo>/data/DR
./scripts/train_dr_vit.sh        # reads <repo>/data/DR automatically
```

**(b) The dataset is already extracted somewhere on this machine** (for example
`/home/jupyter-kl3nguye/.cache/huggingface/...` still holds the archives, or you
find a folder that already contains `DR/Training`, `DR/Validation`, `DR/Test`).
There are two ways to use it, both without editing any file:

```bash
# Option 1 — point the scripts at it for this command only:
DATASET_DIR=/home/jupyter-vshein/data/harvard/FairVision ./scripts/train_dr_vit.sh

# Option 2 — make it permanent by symlinking it into the repository.
#            (`data/` is in .gitignore, so the symlink stays local.)
ln -s /home/jupyter-vshein/data/harvard/FairVision data
./scripts/train_dr_vit.sh
```

**(c) The data lives on another account** (e.g. the archives were downloaded by
`jupyter-kl3nguye` and you are `jupyter-vshein`). If the archives are readable
they can be extracted with

```bash
./scripts/prepare_data.sh DR /home/jupyter-kl3nguye/.cache/huggingface/hub/datasets--harvardairobotics--FairVision/snapshots/<revision>
```

which writes the extracted copy into **your** `<repo>/data/` — no `sudo`, no
ownership problems. Do not try to extract in place: another user's cache is
normally read-only.

The folder you pass to `DATASET_DIR` must be the one that *contains* the
disease folders, so that `DATASET_DIR/DR/train` (or `DATASET_DIR/DR/Training`)
resolves. Either spelling of the split names works.

`train_*.sh` also honours `RESULT_DIR` if you want the logs and checkpoints
somewhere other than `<repo>/results/`.

### 1.5 Compatibility notes for this checkout

Three small fixes were needed so that the (2024) code base runs with the current
PyTorch/timm releases — all of them are already applied in this repository:

1. `scripts/models_vit.py` — the vendored MAE `VisionTransformer` inherits the
   modern timm class, whose `forward()` calls `forward_head()`, pooling and
   normalising the features a second time. An explicit
   `forward()` (features → head) was added, matching the original
   MAE/timm-0.5 behaviour.
2. `requirements.txt` — `timm` is pinned to `>=0.9.3,<1.0.0` (the API the
   repository was written against; the deprecated `timm.models.layers` import is
   still available there).
3. All `scripts/train_*.py` — `torch.cuda.synchronize()` is now wrapped in
   `if torch.cuda.is_available():` and the hard-coded `.cuda()` calls use the
   `device` variable, so the code also runs (slowly) on a CPU-only machine.

---

## 2. Preparing the data

### 2.1 Where the dataset lives

The dataset is published on Hugging Face as
[`harvardairobotics/FairVision`](https://huggingface.co/datasets/harvardairobotics/FairVision).
If it was downloaded before (for example by the `hf download` command below, or
by the account that owns the data), it sits in that account's Hugging Face cache
as one archive per disease:

```text
<$HF_HOME or ~/.cache/huggingface>/hub/datasets--harvardairobotics--FairVision/
└── snapshots/<revision>/
    ├── AMD/
    │   ├── Dataset/dataset.zip          # ~40 GB archive: NPZ files + SLO jpgs
    │   └── ReadMe/{data_description_amd.txt, data_summary_amd.csv}
    ├── DR/
    │   ├── Dataset/dataset.zip
    │   └── ReadMe/{data_description_dr.txt, data_summary_dr.csv}
    └── Glaucoma/
        ├── Dataset/dataset.zip
        └── ReadMe/{data_description_glaucoma.txt, data_summary_glaucoma.csv}
```

The cache only holds the **zipped** archives, so they have to be extracted
before training — that is what `prepare_data.sh` does in the next step.

If the dataset has never been downloaded on this machine:

```bash
hf download harvardairobotics/FairVision --repo-type dataset
# older clients: huggingface-cli download harvardairobotics/FairVision --repo-type dataset
```

You only need to download/extract the disease(s) you plan to train on
(`AMD`, `DR` and/or `Glaucoma`).

### 2.2 Extract the archive(s)

```bash
./scripts/prepare_data.sh DR            # only DR (the default)
./scripts/prepare_data.sh "AMD DR"      # several diseases in one go
```

The script:

* searches your own Hugging Face cache (`$HF_CACHE`, `$HF_HOME`,
  `~/.cache/huggingface`, then any `/home/*/.cache/huggingface`) for the
  snapshot — run it as the user who downloaded the data and no argument is
  needed;
* extracts into `<repo>/data` (skipped automatically if the data is already
  there, so it is safe to re-run);
* adds the lowercase `train` / `val` / `test` aliases next to the archive's
  `Training` / `Validation` / `Test` folders;
* copies `data_summary_<disease>.csv` next to them (race, gender, ethnicity,
  age, preferred language, marital status).

Overrides, if needed:

```bash
HF_CACHE=/path/to/.cache/huggingface \   # where the download lives
DATA_ROOT=/path/to/extract          \   # where to put the extracted data
./scripts/prepare_data.sh DR
```

### 2.3 Resulting layout

```text
<repo>/data/
└── DR/
    ├── Training/            # extracted from dataset.zip
    ├── Validation/
    ├── Test/
    ├── train -> Training    # convenience aliases
    ├── val   -> Validation
    ├── test  -> Test
    └── data_summary_dr.csv
```

The loader (`src/data_handler.py`) accepts either spelling — `Training` /
`Validation` / `Test` work just as well as `train` / `val` / `test` — and only
reads the `*.npz` files (`data_xxxxx.npz`). The `slo_xxxxx.jpg` files are
provided for visual inspection and are not read by the training loop.

---

## 3. Experiments

Run every command from the repository root. Each run writes its log,
`args_train.txt`, the best checkpoints (`model_best_epoch*.pth`) and
`pred_gt_best_epoch*.npz` into `--result_dir` (by default `./results/<name>/`),
and appends one summary row per run to `./results/best_<perf_file>.csv`.

### Step 1 — Baseline ViT-B on DR with SLO fundus images

```bash
./scripts/train_dr_vit.sh
```

* `ViT-B/16` initialised from ImageNet weights, 224×224 SLO fundus images,
  BCE loss for vision-threatening DR (label 1 = severe NPDR / PDR).
* 50 epochs, batch size 64, base LR `5e-4`, 5 warm-up epochs, layer-wise LR
  decay `0.55`, stochastic depth `0.1`.
* Fairness is reported for the `race`, `gender` and `hispanic` attributes.
* Outputs: `./results/DR_ViT-B_slo_fundus_race/` and
  `./results/best_DR_ViT-B_slo_fundus_race.csv`.

Quick smoke test (1 epoch on 1 % of the training data) to check that data,
model and pretrained weights all load correctly:

```bash
python scripts/train_dr_fair.py \
    --epochs 1 --dataset_proportion 0.02 --batch_size 32 --workers 8 \
    --data_dir data/DR/ \
    --result_dir ./results/smoke_dr --model_type ViT-B \
    --modality_types slo_fundus --vit_weights imagenet --attribute_type race
```

### Step 2 — ViT-B + Fair Identity Scaling (FIS) on DR

```bash
./scripts/train_dr_vit_fis.sh
```

Same configuration as Step 1 with the proposed FIS loss enabled
(`--fair_scaling_coef 0.5`, Sinkhorn blur `0.1`, temperature `1.0`).

### Step 3 — Baseline 3D ResNet on DR with OCT B-scans

```bash
./scripts/train_dr_3d.sh
```

3D ResNet-18 (`--conv_type Conv3d`) on OCT volumes (`--image_size 200`),
batch size 2.

### Step 4 — 3D ResNet + FIS on DR with OCT B-scans

```bash
./scripts/train_dr_3d_fis.sh
```

### Running the same four steps on AMD

```bash
./scripts/train_amd_vit.sh       # Step 1 on AMD
./scripts/train_amd_vit_fis.sh   # Step 2 on AMD
./scripts/train_amd_3d.sh        # Step 3 on AMD
./scripts/train_amd_3d_fis.sh    # Step 4 on AMD
```

### Running the same four steps on Glaucoma

Extract the archive first (`./scripts/prepare_data.sh Glaucoma`), then copy one
of the `train_dr_*.sh` scripts and replace `DR` with `Glaucoma` and
`train_dr_fair*` with `train_glaucoma_fair*`.

### Command-line options worth knowing

| Flag | Meaning |
| --- | --- |
| `--data_dir` | folder containing `train/`, `val/`, `test/` for one disease |
| `--attribute_type` | protected attribute to evaluate: `race`, `gender`, `hispanic` |
| `--model_type` | `ViT-B`, `vit`, `resnet`, `convnext`, … |
| `--modality_types` | `slo_fundus` (2D) or `oct_bscans` / `oct_bscans_3d` (3D) |
| `--dataset_proportion` | fraction of the training set to use (smoke tests) |
| `--workers` | number of DataLoader worker processes |
| `--seed` | random seed; the result folder is renamed to include the seed |

---

## 4. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `FileNotFoundError: .../DR/train` | The archive was not extracted yet — run `./scripts/prepare_data.sh DR`. |
| `CUDA out of memory` | Lower `--batch_size` (ViT-B) or `--fair_scaling_batchsize` (3D / FIS). |
| Downloading the ImageNet weights fails | Use `--vit_weights scratch`, or run once on a machine with internet access (timm caches the weights in `~/.cache/huggingface/`). |
| `--vit_weights mae / mocov3 / mae_chest_xray / mae_color_fundus` fails | Those options expect pre-trained checkpoints in the authors' model hub folder. Put the `.pth` files in a folder of your own and point the code at it with the `MODELHUB_DIR` environment variable: `MODELHUB_DIR=/path/to/MODELHUB ./scripts/train_dr_vit.sh` (defaults to the original `/scratch/mok232/...` path, so nothing breaks if you never use these options). Alternatively use `imagenet` or `scratch`. |
| `ImportError` / `ModuleNotFoundError` for a package | Re-run `pip install -r requirements.txt` and check that you run the scripts with the same interpreter (`python -c "import <package>"`). |
| Training is extremely slow | Check `python -c "import torch; print(torch.cuda.is_available())"`. If it prints `False`, no GPU is visible and everything runs on the CPU. |
| `unzip: command not found` | Use `python -m zipfile -e <archive> <target>` instead, or install `unzip` without `sudo` via `conda install -c conda-forge unzip`. |

---

## Acknowledgment and Citation

If you find this repository useful for your research, please consider citing our [paper](https://arxiv.org/pdf/2310.02492):

```bibtex
@misc{luo2024fairvisionequitabledeeplearning,
      title={FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling}, 
      author={Yan Luo and Muhammad Osama Khan and Yu Tian and Min Shi and Zehao Dou and Tobias Elze and Yi Fang and Mengyu Wang},
      year={2024},
      eprint={2310.02492},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2310.02492}, 
}

```


