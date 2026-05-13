import json
import matplotlib.pyplot as plt

# Load JSON file
with open("training_history_gan.json", "r") as f:
    history = json.load(f)

# Extract data
accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(1, len(accuracy) + 1)
import matplotlib.pyplot as plt

# Create side-by-side figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---------------- Accuracy Plot ----------------
axes[0].plot(epochs, accuracy, label='Training Accuracy')
axes[0].plot(epochs, val_accuracy, label='Validation Accuracy')

axes[0].set_title('Training vs Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# ---------------- Loss Plot ----------------
axes[1].plot(epochs, loss, label='Training Loss')
axes[1].plot(epochs, val_loss, label='Validation Loss')

axes[1].set_title('Training vs Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

# Adjust layout
plt.tight_layout()

# Save combined figure
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')

# Show figure
plt.show()

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

n_synthetic = 5
latent_dim = 64
generator = keras.models.load_model("generator.keras")
noise = np.random.normal(0, 1, (n_synthetic, latent_dim))
synthetic_imgs = generator.predict(noise)
synthetic_imgs = (synthetic_imgs + 1) / 2

fig, axes = plt.subplots(1, 5, figsize=(16, 3))

for i in range(5):
    axes[i].imshow(synthetic_imgs[i])
    axes[i].axis("off")
    axes[i].set_title(f"Image {i+1}")

plt.tight_layout()
plt.savefig("generated_images.png")
plt.show()