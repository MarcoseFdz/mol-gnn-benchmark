import time
import os
import numpy as np
import tensorflow as tf
from src.dataset import tf_random_augment

def train_model(
    model, x_train, adj_train, y_train, x_val, adj_val, y_val,
    lr=0.005, epochs=1000, patience=100, batch_size=32, model_name="model",
    use_augmentation=True
):
    def map_augment(inputs, y):
        x, adj = inputs
        if use_augmentation:
            x, adj = tf_random_augment(x, adj)
        return (x, adj), y

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(((x_train, adj_train), y_train))
        .shuffle(len(y_train))
        .map(map_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices(((x_val, adj_val), y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    counts  = np.bincount(y_train.astype(int))
    total   = float(len(y_train))
    class_w = {i: total / (2.0 * c) for i, c in enumerate(counts)}

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc", from_logits=True)],
    )

    checkpoint_path = f"models/checkpoints/best_{model_name}_bace.weights.h5"
    os.makedirs("models/checkpoints", exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            patience=patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_w,
        verbose=0,
    )
    
    final_path = f"models/checkpoints/final_{model_name}_bace.weights.h5"
    model.save_weights(final_path)
    
    print(f"  * Checkpoint saved to: {checkpoint_path}")
    print(f"  * Final weights saved to: {final_path}")

    return history.history
