# Sign Language Recognition using CNN

A deep learning project that uses **Convolutional Neural Networks (CNN)** to recognize static hand gestures representing the American Sign Language (ASL) alphabets (A–Z, excluding J and Z). This system aims to enhance communication accessibility for the speech and hearing-impaired community.

---

## 📌 Features

- 🧠 Built using a custom Convolutional Neural Network
- 📷 Supports **static image recognition** of hand gestures
- 🅰️ Predicts 24 letters of the English alphabet (except J and Z)
- 🔍 Preprocessing includes grayscale conversion, resizing, and normalization
- 📊 Includes training, testing, and performance evaluation
- 📈 Confusion matrix and accuracy/loss plots for visual insights

---

## 🛠️ Tech Stack

| Component        | Technology            |
|------------------|------------------------|
| Programming      | Python                |
| Deep Learning    | TensorFlow, Keras     |
| Data Handling    | NumPy                 |
| Visualization    | Matplotlib, Seaborn   |
| Environment      | Jupyter Notebook / Google Colab |

---

## 📁 Dataset

- The project uses a publicly available **ASL static sign language image dataset**.  
- 📥 Download from: [https://acesse.one/slr-dataset](https://acesse.one/slr-dataset)

### Dataset Highlights
- Contains labeled images for A–Z (excluding J and Z)
- Images are preprocessed to:
  - 📏 Resize to 28x28 pixels
  - 🌑 Convert to grayscale
  - 📊 Normalize pixel values to 0–1 range

---

## 🚀 How It Works

### 🔹 Step 1: Data Loading & Preprocessing
- Load dataset images and labels
- Resize all images to `28x28`
- Convert to grayscale
- Normalize pixel values using division by 255
- Perform train-test split using `train_test_split`

### 🔹 Step 2: Model Building
A Convolutional Neural Network (CNN) is created using TensorFlow/Keras:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(24, activation='softmax')  # 24 classes for A–Z (excluding J, Z)
])
```

### 🔹 Step 3: Training
The model is compiled and trained using the following configuration:

Loss Function: categorical_crossentropy

Optimizer: adam

Metrics: accuracy

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64
)
```

### 🔹 Step 4: Evaluation
After training, the model's performance is evaluated using the validation/test set.

``` python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
```

Also includes:

🔍 Confusion Matrix to analyze misclassifications

📉 Accuracy and Loss plots to visualize training and validation trends

### 🔹 Step 5: Prediction
Once trained, you can use the model to predict new hand sign images:

``` python
prediction = model.predict(image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(prediction)
```

---

## 📊 Results
✅ Achieved X% validation accuracy (replace X with your achieved result)

🧾 Confusion matrix highlights model performance per class

📈 Accuracy/loss graphs available for training and validation analysis

---

## 📌 Future Improvements
📹 Real-time hand gesture prediction using webcam

🕹️ Support for dynamic gestures (e.g., J, Z)

🌏 Extend to Indian Sign Language (ISL)

⚙️ Enhance with transfer learning for improved accuracy

---

## 🤝 Contributing
Contributions are welcome!
Feel free to fork this repository, raise issues, or submit pull requests to improve the project.
