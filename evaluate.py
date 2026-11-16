import argparse
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = (224, 224)
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/raw-img")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model_path = "models/animal_mobilenetv2.keras"
    labels_path = "models/class_names.json"
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Train the model first by running: python train.py")

    ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=args.batch_size, label_mode="int", shuffle=False
    )
    source_names = ds.class_names
    class_map = {
        "farfalla": "butterfly", "gatto": "cat", "gallina": "chicken",
        "mucca": "cow", "cane": "dog", "elefante": "elephant",
        "cavallo": "horse", "pecora": "sheep", "ragno": "spider",
        "scoiattolo": "squirrel"
    }
    labels = [class_map[n] for n in source_names]
    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
    probabilities = model.predict(ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    os.makedirs("outputs", exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print("\nClassification report:\n", report)
    with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    display.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Animal classifier confusion matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=160)
    plt.close()
    print("Saved report and confusion matrix in outputs/")


if __name__ == "__main__":
    main()
