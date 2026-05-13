from tensorflow import keras
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# Load model
classifier = keras.models.load_model("classifier.keras")

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

image_size = (32, 32)
image_size_rgb = (*image_size, 3)
print("Loading testing data...")
x_test, y_test = load_dataset('/Volumes/Sugandasu/dataset/wildfire/test', image_size)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_test)
y_train_onehot = to_categorical(y_train_encoded)
X_test, Y_test = shuffle(x_test, y_train_onehot)

prediction = classifier.predict(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(prediction, axis=1)
y_true = np.argmax(Y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Display
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap='Blues', ax=ax)

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')

print(f"Accuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"Precision: {precision:.4f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=le.classes_
))

# Classification report as dictionary
report = classification_report(
    y_true,
    y_pred,
    target_names=le.classes_,
    output_dict=True
)

# Create table for per-class metrics
metrics_table = []

for class_name in le.classes_:
    metrics_table.append({
        "Class": class_name,
        "Precision": report[class_name]["precision"],
        "Recall": report[class_name]["recall"],
        "F1-Score": report[class_name]["f1-score"],
        "Support": report[class_name]["support"]
    })

# Add overall metrics
metrics_table.append({
    "Class": "Overall",
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Support": len(y_true)
})

# Convert to DataFrame
df_metrics = pd.DataFrame(metrics_table)

# Display table
print("\nPer-Class Metrics:")
print(df_metrics)

# Save to CSV
df_metrics.to_csv("classification_metrics.csv", index=False)

# Optional: save as formatted image/table
fig, ax = plt.subplots(figsize=(8, len(df_metrics) * 0.6 + 1))
ax.axis('off')

table = ax.table(
    cellText=df_metrics.round(4).values,
    colLabels=df_metrics.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Classification Metrics Per Class", pad=20)
plt.savefig("classification_metrics.png", bbox_inches='tight', dpi=300)
plt.show()

from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Create table
class_accuracy_table = []

for i, class_name in enumerate(le.classes_):
    class_accuracy_table.append({
        "Class": class_name,
        "Accuracy": class_accuracy[i]
    })

# Overall accuracy
overall_accuracy = np.trace(cm) / np.sum(cm)

class_accuracy_table.append({
    "Class": "Overall",
    "Accuracy": overall_accuracy
})

# Convert to DataFrame
df_accuracy = pd.DataFrame(class_accuracy_table)

# Display
print(df_accuracy)

# Save CSV
df_accuracy.to_csv("class_accuracy.csv", index=False)