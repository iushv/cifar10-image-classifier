# CIFAR-10 Image Classifier

A CNN that classifies small images into 10 categories. Built to understand how convolutional networks actually learn.

---

## What It Does

Takes a 32×32 color image and tells you if it's an airplane, car, bird, cat, deer, dog, frog, horse, ship, or truck.

This is the standard benchmark everyone uses when learning CNNs. I built this to understand batch normalization, dropout, and how the pieces fit together—not just to call `model.fit()`.

---

## Results

- **Test Accuracy:** ~85%
- **Training Time:** ~10 minutes on GPU
- **Model Size:** 6.5 MB

The confusion matrix shows the model struggles most with cats vs dogs (understandably—they're both fuzzy mammals at 32×32 pixels).

---

## Running It

### Jupyter Notebook (Recommended)

```bash
git clone https://github.com/iushv/cifar10-image-classifier.git
cd cifar10-image-classifier

# Local
jupyter notebook cifar10_image_classifier.ipynb

# Or upload to Google Colab and click Run All
```

### Python Script

```bash
pip install -r requirements.txt
python cifar10_classifier.py
```

---

## The Model

Three convolutional blocks, each with:
- 2 Conv layers (32 → 64 → 128 filters)
- Batch normalization after each conv
- 2×2 max pooling
- Dropout (25%)

Then a dense layer with 50% dropout, and softmax output.

Total: ~550K parameters. Small enough to train on a laptop.

---

## What I Learned

**Batch normalization matters.** Without it, training was unstable and took 3x longer to converge.

**Dropout prevents overfitting.** The dataset is small (60K images). Without dropout, training accuracy hit 100% while test stayed at 75%.

**Data augmentation would help.** I didn't add it here, but random flips and rotations would probably push accuracy to ~90%.

---

## Files

```
cifar10-image-classifier/
├── cifar10_image_classifier.ipynb   # Main notebook
├── cifar10_classifier.py            # Standalone script
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **TensorFlow/Keras** — Model building and training
- **Matplotlib/Seaborn** — Visualizations
- **scikit-learn** — Confusion matrix and metrics

---

Built as a learning exercise. The classics matter before you chase transformers.
