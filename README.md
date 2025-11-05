# Brain Tumor Classification Using Deep Learning

This project classifies MRI brain images into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

Three deep learning models are implemented and compared:
1. **Custom CNN** (built from scratch)
2. **Multi-Layer Perceptron (MLP)**
3. **VGG16 Transfer Learning** (pre-trained on ImageNet)

The training and evaluation are performed in **Google Colab** using GPU acceleration.

---

## Dataset

- **Source:** Brain Tumor MRI Dataset (Kaggle)
- **Training Images:** ~5,712  
- **Testing Images:** ~656  

Directory structure:
Training/
├── glioma
├── meningioma
├── notumor
└── pituitary

Testing/
├── glioma
├── meningioma
├── notumor
└── pituitary

yaml
Copy code

---

## Preprocessing & Augmentation
- Image resized to **150×150**
- Normalized to `[0,1]`
- Data augmentation:
  - Rotation
  - Width/Height Shift
  - Zoom
  
---

## Models & Results

| Model | Description | Test Accuracy | Macro F1-Score |
|------|-------------|--------------|----------------|
| **Custom CNN** | 3 Conv Blocks + Dense Layers | **88%** | **0.87** |
| **MLP** | Flattened Input → Dense Layers | **82%** | **0.80** |
| **VGG16 Transfer Learning** | Frozen base + Custom Classifier Head | **96%** | **0.96** |

> **Conclusion:** VGG16 clearly outperforms the custom models with higher precision and recall across all classes.

---

## Requirements

tensorflow>=2.15.0
keras
numpy
pandas
scikit-learn
matplotlib
seaborn

go
Copy code

Install:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
How to Run
1. Open the notebook in Google Colab
2. Enable GPU:
bash
Copy code
Runtime → Change runtime type → GPU
3. Mount Google Drive:
python
Copy code
from google.colab import drive
drive.mount('/content/drive')
4. Run all cells sequentially.
Training time: 5–10 minutes/model on GPU.

Predict on a New Image
python
Copy code
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('VGG_model.h5')

img = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(150,150))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
pred = model.predict(np.expand_dims(img_array, axis=0))

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classes[np.argmax(pred)])
Future Improvements
Handle class imbalance using weighted loss / oversampling

Expand augmentation (brightness, flips, elastic transforms)

Model interpretability (Grad-CAM visualizations)

Deploy via Flask / FastAPI / TensorFlow Lite

License
MIT License
Feel free to use, improve, and distribute.
