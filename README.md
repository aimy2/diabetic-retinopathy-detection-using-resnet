# 🩺 Diabetic Retinopathy Detection using Deep Learning  
**GDGOC PIEAS AI/ML Hackathon 2025**

---

## 📌 Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that can lead to blindness if not detected early.  
This project implements a **custom ResNet-style CNN trained from scratch** (no pretrained weights) to classify retinal fundus images into **five DR severity levels**, along with **Grad-CAM explainability** to visualize clinically relevant regions.

---

## 🎯 Key Features

- ✅ Fully original CNN (no pretrained weights)
- ✅ GPU-supported training (RTX 4060 compatible)
- ✅ Five-class DR severity classification
- ✅ Grad-CAM visual explainability
- ✅ Hackathon-rule compliant

---

## 🧠 Model Architecture

- Custom **ResNet-style CNN**
- Residual connections for stable training
- Global Average Pooling
- MLP classifier head

### DR Severity Classes

| Label | Class Name |
|-----|-----------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

---

## 📂 Dataset

Dataset used:  
🔗 https://www.kaggle.com/datasets/kushagratandon12/diabetic-retinopathy-balanced/data

### Folder Structure
Diabetic_Balanced_Data/
├── train/
│ ├── 0/
│ ├── 1/
│ ├── 2/
│ ├── 3/
│ ├── 4/
├── test/
│ ├── 0/
│ ├── 1/
│ ├── 2/
│ ├── 3/
│ ├── 4/

---

## ⚙️ Installation

### 1️⃣ Create environment (recommended)
```bash
conda create -n llasa python=3.9 -y
conda activate llasa
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


Training the Model

Run the training script:

python trainingscript.py


This will:

Train the ResNet model from scratch

Evaluate on the test set

Save the best model as:

best_dr_resnet.pt

🔍 Explainability with Grad-CAM

Grad-CAM highlights the retinal regions influencing the model’s prediction.

Run Grad-CAM:
python gradcam_resnet_dr.py

Output:

Console: predicted class + confidence

Image file:

gradcam_output.jpg

📊 Baseline Results

Weighted F1-score: ~0.70

Strong performance on Severe and Proliferative DR

Moderate confusion between No / Mild / Moderate DR (expected)

🧪 Explainability

Grad-CAM visualizations highlight:

Hemorrhages

Lesions

Pathological retinal regions

This improves interpretability and clinical trust.