import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, BatchNormalization, Reshape, Conv2DTranspose, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

def load_dataset(dir_path, img_size=(32, 32)):
    x = []
    y = []
    for direct in os.listdir(dir_path):
        fullpath = os.path.join(dir_path, direct)
        if not os.path.isdir(fullpath):
            continue
        print(f"Loading dataset from {direct}")
        for filename in os.listdir(fullpath):
            img_path = os.path.join(dir_path, direct, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = np.array(img) / 255.0
            x.append(img)
            y.append(direct)
    return np.array(x), y

def build_generator(latent_dim=64):
    model = Sequential([
        Input(shape=(latent_dim,)),
        Dense(8 * 8 * 256, activation='relu'),
        Reshape((8, 8, 256)),

        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2D(3, (3, 3), padding='same', activation='tanh')
    ], name='generator')

    return model

def build_discriminator(img_shape=(32, 32, 3)):
    model = Sequential([
        Input(shape=img_shape),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),

        Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Dropout(0.3),

        Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Dropout(0.3),

        Flatten(),
        Dense(1, activation='sigmoid')
    ], name='discriminator')

    return model

def train_gan(X_train, epochs=50, batch_size=32, latent_dim=64, save_interval=10):
    # Normalize to [-1, 1] for tanh activation
    X_train_gan = (X_train * 2) - 1

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    history = {'d_loss': [], 'g_loss': [], 'd_acc': [], 'g_acc': []}

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train_gan.shape[0], batch_size)
        real_imgs = X_train_gan[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        discriminator.trainable = False
        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        history['d_loss'].append(d_loss[0])
        history['d_acc'].append(d_loss[1])
        history['g_loss'].append(g_loss[0])
        history['g_acc'].append(g_loss[1])

        if epoch % save_interval == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss[0]:.4f},")

    return history

image_size = (32, 32)
image_size_rgb = (*image_size, 3)

print("Loading training data...")
x_train, y_train = load_dataset('/Volumes/Sugandasu/dataset/wildfire/train', image_size)

print("Loading validation data...")
x_val, y_val = load_dataset('/Volumes/Sugandasu/dataset/wildfire/valid', image_size)

# Encode label training, then shuffle data
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_train_onehot = to_categorical(y_train_encoded)
X_train, Y_train = shuffle(x_train, y_train_onehot)

# Encode label validation, then shuffle data
y_val_encoded = le.transform(y_val)
y_val_onehot = to_categorical(y_val_encoded)
X_val, Y_val = shuffle(x_val, y_val_onehot)

# Display file size
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

latent_dim = 64

# Build and compile generator
generator = build_generator(latent_dim)

# Build and compile discriminator
discriminator = build_discriminator(image_size_rgb)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False

# Build GAN model
gan_input = Input(shape=(latent_dim,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)
gan = Model(gan_input, gan_output, name='gan')
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
generator.summary()
discriminator.summary()

# Training gan
print("\nTraining gan")
gan_history = train_gan(X_train, epochs=100, batch_size=32, latent_dim=latent_dim, save_interval=10)

generator.save("generator.keras")
discriminator.save("discriminator.keras")
gan.save("gan.keras")

# Generate synthetic images to augment dataset
ratio = 0.2
n_synthetic = int(X_train.shape[0] * ratio)

noise = np.random.normal(0, 1, (n_synthetic, latent_dim))
synthetic_imgs = generator.predict(noise)
synthetic_imgs = (synthetic_imgs + 1) / 2

X_train_augmented = np.concatenate([X_train, synthetic_imgs])
Y_train_augmented = np.concatenate([
    Y_train,
    np.tile([1, 0], (n_synthetic, 1))
])

print(f"\nAugmented training samples: {X_train_augmented.shape[0]}")

# Build and compile CNN classifier
classifier = Sequential([
    Input(shape=image_size_rgb),
    Conv2D(32, 3, padding="same", activation="relu"),
    Conv2D(64, 3, padding="same", activation="relu"),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(2, activation="softmax")
], name='classifier')

classifier.summary()

classifier.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train classifier with augmented data
print("\nTraining classifier")
history = classifier.fit(
    X_train_augmented, Y_train_augmented,
    validation_data=(X_val, Y_val),
    batch_size=32,
    epochs=10,
    verbose=1
)

classifier.save("classifier.keras")

with open("training_history_gan.json", "w") as f:
    json.dump(history.history, f)

# Plot GAN training
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(gan_history['d_loss'], label='Discriminator Loss')
plt.plot(gan_history['g_loss'], label='Generator Loss')
plt.title('GAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot GAN accuracy
plt.subplot(1, 2, 2)
plt.plot(gan_history['d_acc'], label='Discriminator Accuracy')
plt.plot(gan_history['g_acc'], label='Generator Accuracy')
plt.title('GAN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('gan_results.png')
plt.show()

# Plot classifier accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Classifier Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('classifier_accuracy.png')
plt.legend()

# Plot classifier loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Classifier Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('classifier_loss.png')
plt.legend()

