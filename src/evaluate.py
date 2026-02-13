
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data and prints metrics."""
    print("Evaluating model...")
    y_pred = model.predict(X_test).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Evaluation Results:")
    print(f"✅ Test RMSE: {rmse:.4f}")
    print(f"✅ Test R2 Score: {r2:.4f}")
    
    return rmse, r2, y_pred

def plot_history(history):
    """Plots training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    # Save plot instead of showing it (better for scripts)
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")

def plot_predictions(y_true, y_pred, n_samples=100):
    """Plots actual vs predicted RUL for a subset of samples."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:n_samples], label='Actual RUL')
    plt.plot(y_pred[:n_samples], label='Predicted RUL')
    plt.title(f'Actual vs Predicted RUL (First {n_samples} Samples)')
    plt.xlabel('Sample')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True)
    plt.savefig('predictions_plot.png')
    print("Predictions plot saved to predictions_plot.png")
