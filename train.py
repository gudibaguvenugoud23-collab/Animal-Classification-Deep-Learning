import argparse
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt

CLASS_MAP = {
    "farfalla": "butterfly",
    "gatto": "cat",
    "gallina": "chicken",
    "mucca": "cow",
    "cane": "dog",
    "elefante": "elephant",
    "cavallo": "horse",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Train the 10-class animal classifier.")
    parser.add_argument("--data_dir", default="dataset/raw-img", help="Folder containing the 10 class subfolders.")
    parser.add_argument("--epochs", type=int, default=8, help="Frozen-backbone training epochs.")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {args.data_dir}\n"
            "Download Animals-10, extract it, and point --data_dir to its raw-img folder."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=args.batch_size, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=args.batch_size, label_mode="int"
    )

    source_names = train_ds.class_names
    missing = sorted(set(CLASS_MAP) - set(source_names))
    if missing:
        raise ValueError(f"Expected Animals-10 folders are missing: {missing}. Found: {source_names}")
    english_names = [CLASS_MAP[name] for name in source_names]
    print("Class index mapping:", dict(enumerate(english_names)))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
    ], name="data_augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(english_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/animal_mobilenetv2.keras", monitor="val_accuracy",
            save_best_only=True, mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7
        ),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    print("\nStage 1: training the classification head")
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    print("\nStage 2: fine-tuning upper MobileNetV2 layers")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    # Keep Batch Normalization layers frozen for stable fine-tuning on smaller datasets.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    hist2 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
        callbacks=callbacks
    )

    # Save the best checkpoint and the exact class order used by the model.
    best_model = tf.keras.models.load_model("models/animal_mobilenetv2.keras")
    best_model.save("models/animal_mobilenetv2.keras")
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(english_names, f, indent=2)

    histories = {}
    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        histories[key] = hist1.history.get(key, []) + hist2.history.get(key, [])
    with open("outputs/training_history.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    epochs_range = range(1, len(histories["accuracy"]) + 1)
    plt.figure()
    plt.plot(epochs_range, histories["accuracy"], label="Training accuracy")
    plt.plot(epochs_range, histories["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/accuracy.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs_range, histories["loss"], label="Training loss")
    plt.plot(epochs_range, histories["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/loss.png", dpi=150)
    plt.close()

    loss, accuracy = best_model.evaluate(val_ds, verbose=0)
    print(f"\nBest checkpoint validation loss: {loss:.4f}")
    print(f"Best checkpoint validation accuracy: {accuracy:.4f}")
    print("Saved model: models/animal_mobilenetv2.keras")
    print("Saved labels: models/class_names.json")


if __name__ == "__main__":
    main()
