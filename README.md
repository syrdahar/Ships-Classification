# Ship Classification — Hand-Written CNN

Ten-class classification of greyscale ship images (128×192 px) with a convolutional
network written from scratch in PyTorch, under an assignment constraint of fewer
than 30 layers and no pre-trained backbone. The project goes from a plain VGG-style
baseline to a densified architecture combined in a multi-seed deep ensemble, with a
systematic study of regularisation, test-time augmentation and ensembling along the way.

## Repository structure

```
.
├── ships_classification.ipynb   # full pipeline: data, training, ablations, submission
├── rapport.pdf/                 # compiled report, 13 pages
└── README.md
```

The dataset is not versioned. The notebook expects it at `./ships24/`:

```
ships24/
├── ships_gray/ships_gray/<class_name>/*.jpg   # 37,980 labelled images, 10 classes
└── test_images.npy                            # 4,224 unlabelled competition images
```

## Getting started

```bash
pip install torch torchvision numpy pandas pillow matplotlib seaborn scikit-learn
jupyter notebook ships_classification.ipynb
```

Developed with Python 3.10 and CUDA. A GPU is strongly recommended: the notebook uses
a batch size of 1024 with `bfloat16` mixed precision, and the full run (baseline,
four ablations, two v2 variants and five ensemble seeds) takes several hours on an
A100. On CPU, mixed precision is disabled automatically and the batch size should be
reduced.

Running the notebook end to end writes `test.csv` (`ID,Category`) for submission.

## Technologies

PyTorch · torchvision · NumPy · pandas · Pillow · scikit-learn (stratified split,
confusion matrix, classification report) · matplotlib · seaborn

## Results

Final standing: **2nd out of 71** on the competition leaderboard.

The accuracies below are measured on an internal stratified test split of 3,798
images, held out from training and model selection.

| Configuration | Params | Test accuracy |
|---|---:|---:|
| `DeepGAPCNN` baseline | 4.7M | 92.21% |
| \+ label smoothing (best single v1) | 4.7M | 92.92% |
| Ensemble of 4 v1 models | 4×4.7M | 93.42% |
| `DeepGAPCNN_v2` | 12.2M | 94.97% |
| `DeepGAPCNN_v2` \+ strong aug \+ MixUp | 12.2M | 96.92% |
| **5-seed ensemble of v2** | 5×12.2M | **97.39%** |

The headline finding is that strong augmentation, MixUp and CutMix all *hurt* the
4.7M-parameter baseline (by 1.6 to 4.2 points) and only become profitable once the
network is densified — the model was bias-limited, not variance-limited.

Full methodology, per-class analysis, ablations, diversity study and limitations:
**[`rapport.pdf`](rapport.pdf)**.
