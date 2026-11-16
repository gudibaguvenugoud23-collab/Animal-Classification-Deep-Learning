# Animal Classification and Prediction Using Deep Learning

An advanced 10-class image-classification project using **MobileNetV2 transfer learning**, TensorFlow/Keras, and a Flask web interface.

## 1. Animal classes

The project uses the Animals-10 dataset and these 10 classes:

| English class | Original Animals-10 folder |
|---|---|
| butterfly | farfalla |
| cat | gatto |
| chicken | gallina |
| cow | mucca |
| dog | cane |
| elephant | elefante |
| horse | cavallo |
| sheep | pecora |
| spider | ragno |
| squirrel | scoiattolo |

## 2. Dataset setup

Download the **Animals-10** dataset from Kaggle:
https://www.kaggle.com/datasets/alessiocorrado99/animals10

Extract it. The archive commonly contains a folder named `raw-img` with one subfolder per class.

Arrange the project like this:

```text
animal_classification_dl/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── uploads/                 # created automatically
├── models/                  # created after training
├── outputs/                 # created after evaluation
└── dataset/
    └── raw-img/
        ├── cane/
        ├── cavallo/
        ├── elefante/
        ├── farfalla/
        ├── gallina/
        ├── gatto/
        ├── mucca/
        ├── pecora/
        ├── ragno/
        └── scoiattolo/
```

You can alternatively place the extracted `raw-img` folder anywhere and pass its path to the training script.

## 3. Setup (Windows PowerShell)

Open VS Code terminal in the project folder:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Use a Python version supported by your TensorFlow release (Python 3.11 is a practical choice). Avoid assuming the newest Python release is supported by every TensorFlow wheel.

## 4. Train the model

With the dataset at `dataset/raw-img`:

```powershell
python train.py --data_dir dataset/raw-img --epochs 8 --fine_tune_epochs 5
```

The script:
- Creates a reproducible 80/20 train/validation split.
- Resizes images to 224 x 224.
- Applies training-only data augmentation.
- Uses ImageNet-pretrained MobileNetV2 as a feature extractor.
- Fine-tunes the upper layers with a low learning rate.
- Saves the best model to `models/animal_mobilenetv2.keras`.
- Saves class names to `models/class_names.json`.
- Saves accuracy/loss plots to `outputs/`.

The first run needs internet access to download MobileNetV2 ImageNet weights. For a GPU, install the TensorFlow build appropriate for your operating system and hardware.

## 5. Evaluate

```powershell
python evaluate.py --data_dir dataset/raw-img
```

This evaluates the saved model on a held-out split and writes a classification report and confusion matrix into `outputs/`. The evaluation split uses the same deterministic split seed as training.

## 6. Run the web application

After training:

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser. Upload a JPG, JPEG, or PNG image. The app shows the top-3 predicted classes and confidence scores.

## 7. Project workflow

1. Dataset collection: labelled animal images from Animals-10.
2. Preprocessing: decode, resize, normalize with MobileNetV2 preprocessing.
3. Augmentation: random horizontal flip, rotation, and zoom during training.
4. Feature extraction: pretrained MobileNetV2 convolutional base.
5. Classification: global average pooling, dropout, and 10-unit softmax layer.
6. Fine-tuning: unfreeze only the upper backbone layers and train with a small learning rate.
7. Evaluation: accuracy, precision, recall, F1-score, and confusion matrix.
8. Deployment: Flask accepts an image and returns the top-3 model predictions.

## 8. Important notes

- This is a **single-label image classifier**, not an object detector. It predicts the dominant animal class in one image.
- Softmax confidence is a model score, not a guarantee. An unrelated image may still receive a high score; the current app does not claim reliable out-of-distribution detection.
- Dataset quality, class imbalance, and image leakage can affect results. Review the dataset and report the measured results from your own training run; do not invent accuracy values.
- Animals-10 images are for academic experimentation. Check the dataset page and image licenses/terms before redistributing images.
