
import tensorflow as tf
import os
from src.config import BATCH_SIZE, EPOCHS, MODEL_PATH

def train_model(model, X_train, y_train, validation_split=0.2):
    """Trains the model with EarlyStopping and saves the best version."""
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )
    
    # Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return history
