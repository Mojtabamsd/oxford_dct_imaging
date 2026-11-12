# Teaching machines to see plankton

**Learning from imbalanced data, testing on the unseen.**
A team challenge for the Oxford Doctoral Training Centre, set by the National Oceanography Centre (NOC).

Underwater Vision Profilers (UVPs) photograph particles and plankton as they drift through the ocean.
Machine-learning classifiers turn millions of these images into estimates of plankton abundance and
carbon-relevant particle distributions. Classifiers are usually trained on public benchmark data,
but it is rarely clear how well they work on a *new* deployment, especially for rare but
ecologically important classes in heavily imbalanced data.

Your job: train classifiers on the public **UVP6Net** dataset (634,459 labelled images), then find out
how well they transfer to an independent, expert-labelled **NOC UVP6** deployment whose labels are
hidden from you.

No oceanography background is needed.

---

## Quick start

```bash
git clone https://github.com/Mojtabamsd/oxford_dct_imaging.git
cd oxford_dct_imaging

# 1. Environment (Python 3.10+). On a GPU machine, install PyTorch first: https://pytorch.org/get-started/locally/
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[deep]"

# 2. Data (~1 GB; --data-dir to put it elsewhere, e.g. a scratch disk)
python scripts/download_uvp6net.py
python scripts/download_noc_uvp6.py                     # the NOC test images (no labels)

# 3. A first model: random forest on handcrafted features (CPU, ~2-5 min)
python scripts/train_features.py

# 4. Score it on the UVP6Net test split
python scripts/evaluate.py outputs/rf/predictions_uvp6net_test.csv --truth uvp6net-test --level 1
```

Then upload `outputs/rf/predictions_noc.csv` to the **[leaderboard](https://oxforddctimaging-m5yxepuist23yhhnnfx3tc.streamlit.app/)**
to see how it does on the new deployment.

Check the installation with `pip install -e ".[dev]" && pytest`.

---

## The data

### Training: UVP6Net (public)

| | |
|---|---|
| Images | 634,459 greyscale crops, dark objects on a light background, ~20 px to ~1000 px |
| Labels | **level 1**: 54 classes; **level 2**: 32 coarser classes (each level-1 class belongs to one level-2 class) |
| Splits | fixed `train` / `valid` / `test` (70 / 15 / 15 %, stratified by class) |
| Features | 62 handcrafted morphological features per object (`area`, `feret`, `mean` grey level, ...) |
| Imbalance | `detritus` is **80%** of the images; the rarest classes have ~100 |

`configs/uvp6net_classes.csv` lists every class with its level-2 parent and split counts.
`scripts/download_uvp6net.py --parts data` fetches only the labels and features (~425 MB), which is
enough for feature-based models. The default also fetches the images.

### Testing: NOC UVP6 (labels hidden)

3,586 images from a 2024 NOC UVP6 deployment in the North Atlantic, labelled by an expert taxonomist.
You get the images, the same 62 features, and sampling context (date, position, depth), but **not
the labels**.

* It contains **24 classes**: 23 of the 32 level-2 UVP6Net classes, plus **one class that does not
  exist in the training data**. Your model has never seen it. Label objects you think belong to it
  `unknown`.
* The images were **not sampled at random** from the deployment: rare classes were over-sampled so
  they could be evaluated at all. The leaderboard therefore also reports a *weighted* score, which
  estimates performance on the deployment as a whole.
* It comes from a different deployment, region and time than UVP6Net. Look for differences
  (backgrounds, sizes, class mix) before you train.

### In Python

```python
from plankton.data import load_uvp6net, load_uvp6net_features, load_noc, write_predictions

train = load_uvp6net("train")            # object_id, level1, level2, plankton, split, path
feats = load_uvp6net_features("valid")   # ... plus the 62 features
noc = load_noc()                         # object_id, sample/position/depth, 62 features, path
write_predictions(noc.object_id, my_labels, "outputs/my_model/predictions_noc.csv")
```

---

## Baselines

| Script | What it does | Hardware |
|---|---|---|
| `scripts/train_features.py` | random forest on the 62 features | CPU, minutes |
| `scripts/train_cnn.py` | fine-tunes an ImageNet CNN (ResNet-18 by default) on the images | GPU recommended; CPU with `--max-per-class` |
| `scripts/predict_cnn.py` | predictions from a saved CNN, optionally labelling low-confidence images `unknown` | either |

```bash
# check it runs (CPU ok, ~10 min)
python scripts/train_cnn.py --max-per-class 50 --max-valid 2000 --epochs 1 --img-size 64 --out-dir outputs/cnn_smoke
# baseline (GPU)
python scripts/train_cnn.py --epochs 10 --out-dir outputs/cnn_ce
python scripts/predict_cnn.py outputs/cnn_ce/best.pt --on noc --unknown-threshold 0.5
```

**GPU or CPU, your choice.** The scripts use a GPU automatically when there is one (`--device cpu`
forces CPU). For full-size CNN training, use your university's GPU cluster or a cloud notebook.
Feature-based models run fine on a laptop.

---

## The challenge

Suggested route through the four weeks. Take it in whatever direction your team finds interesting.

**Week 1: explore.** Look at the images of each class. How imbalanced is it, really? How do the NOC
images differ from UVP6Net (sizes, backgrounds, features)? Run both baselines and read their
per-class results: which classes fail, and why?

**Weeks 2-3: build and compare.**
* **Imbalance strategies.** `src/plankton/imbalance.py` has stubs for weighted cross-entropy, focal
  loss, logit adjustment and class-balanced sampling, each with a hint and a reference. Implement
  them, and compare them fairly: same model, same epochs, several seeds.
* **Features vs deep learning**, and combinations of the two. CNN inputs are resized, so they lose
  the absolute object size; the features keep it.
* **The unseen class.** How can a model say "none of the above"? Start from
  `predict_cnn.py --unknown-threshold`, then try something better.
* **Domain shift.** Does performance drop from UVP6Net-test to NOC? For which classes? Can you
  reduce the gap (augmentation, normalisation, using the unlabelled NOC images)?

**Week 4: report.** Per-class comparison of methods, the size of the gap on new data, and practical
recommendations for applying public benchmarks to new ocean data.

> **Tune on `valid`, not on the leaderboard.** You can score the UVP6Net validation split locally as
> often as you like (`--truth uvp6net-valid`). The leaderboard allows a limited number of submissions
> per day and is meant as a final check, not a training signal.

---

## Scoring

`scripts/evaluate.py` and the leaderboard use the same code.

| Metric | Meaning |
|---|---|
| **Macro F1** (ranking) | F1 averaged over classes, each class counting equally, `unknown` included |
| Balanced accuracy | recall averaged over classes |
| Closed-set macro F1 | macro F1 on images of classes seen in training only |
| Unknown recall | share of novel-class images labelled `unknown` |
| False-unknown rate | share of known-class images wrongly labelled `unknown` |
| Weighted macro F1 | corrects for the non-random sampling of the NOC images |

Accuracy is printed too, but it's a trap here: predicting `detritus` for everything scores 80%.

**Predictions format:** a CSV with columns `object_id,label`, one row per image. Labels are
UVP6Net class names at level 1 or level 2 (level 1 is mapped to level 2 for scoring), or `unknown`.

```bash
python scripts/evaluate.py preds.csv --truth uvp6net-valid --level 1 --out-dir results/   # + per-class CSV, confusion matrix
```

### Leaderboard

**<https://oxforddctimaging-m5yxepuist23yhhnnfx3tc.streamlit.app/>**: choose your team, enter your
team code (from the organisers), and upload your `predictions_noc.csv`.

---

## Repository layout

```
configs/uvp6net_classes.csv   class list (level 1 -> level 2, counts)
src/plankton/                 the Python package
  data.py                     load UVP6Net / NOC tables, write predictions
  images.py                   image preprocessing and PyTorch Dataset
  models.py                   CNN architectures, device selection
  imbalance.py                losses and samplers: implement your strategies here
  inference.py                prediction and checkpoints
scripts/                      download data, train baselines, predict, evaluate
tests/                        quick sanity checks (pytest)
leaderboard/                  the leaderboard web app (organisers)
data/, outputs/               created by the scripts, not tracked by git
```

---

## Data licences and citation

* **UVP6Net**: Picheral M., Jalabert L., Motreuil S., Courchet L., Carray-Counil L., Ricour F.,
  Panaïotis T., Petit F., Elineau A. (2024). *UVP6Net: plankton images captured with the UVP6.*
  SEANOE. <https://doi.org/10.17882/101948>. Licence: **CC BY-NC 4.0** (non-commercial use, with attribution).
* **NOC UVP6 test images**: National Oceanography Centre (2026), Zenodo,
  <https://zenodo.org/records/24160827>. Licence: CC BY 4.0.
* Related work: Panaïotis et al. (2026), a benchmark of plankton image classifiers on UVP data,
  *Earth System Science Data*, <https://doi.org/10.5194/essd-18-945-2026>; Masoudi et al. (2024),
  combining imagery from different UVP generations, *IEEE JSTARS*.

The code in this repository is released under the [BSD 3-Clause licence](LICENSE).

**Challenge supervisor:** Mojtaba Masoudi, National Oceanography Centre.
