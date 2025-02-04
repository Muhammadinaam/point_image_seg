# **Technical Report: Semantic Segmentation with Ensemble Learning and Data Amplification**

## **1. Method**
### **1.1 Overview**
This project focuses on **semantic segmentation** using **deep learning** with **semi-supervised learning**, **ensemble models**, and **data amplification**. The model is based on **FCN-ResNet50**, a fully convolutional network trained on the **PASCAL VOC dataset**.

### **1.2 Key Enhancements**
1. **Ensemble Learning**: Trains multiple models and averages their predictions for better accuracy.
2. **Data Amplification**: Uses augmentation techniques (flipping, color jittering) to improve generalization.
3. **Semi-Supervised Learning**: Utilizes sparse point annotations and full segmentation masks.
4. **Progress Tracking**: Uses `tqdm` for real-time training and evaluation progress.
5. **Experimental Analysis**: Evaluates different hyperparameters to determine optimal configurations.

---

## **2. Experimentation**
### **2.1 Purpose & Hypothesis**
We conducted **three major experiments** to analyze how different factors impact segmentation performance.

| **Experiment**      | **Hypothesis** |
|---------------------|---------------|
| **Pretrained vs. Non-Pretrained** | Pretrained models will generalize better and converge faster. |
| **Learning Rate Impact** | A lower learning rate (0.001) will lead to more stable convergence, while higher rates (0.01) may cause instability. |
| **Epoch Count Analysis** | Increasing epochs should improve performance but may lead to diminishing returns after a certain point. |

---

### **2.2 Experimental Process**
Each experiment follows the **same training pipeline** with modifications to specific parameters:

#### **1. Data Preparation & Augmentation**
- **Dataset**: **PASCAL VOC 2012** (automatically downloaded if not available).
- **Transformations**: Images and masks are resized to **256×256**.
- **Amplification**: Horizontal flips, vertical flips, and color jittering are applied.

#### **2. Model Training**
- **Architecture**: FCN-ResNet50 with **21 classes** (for PASCAL VOC).
- **Loss Function**: Custom **Partial Cross-Entropy Loss** (semi-supervised).
- **Optimizer**: Adam optimizer with different **learning rates (0.01, 0.001, 0.0001)**.
- **Training Duration**: Varies across experiments (5, 10, or 20 epochs).
- **Batch Size**: 16.
- **Hardware**: CUDA-enabled GPU (if available), otherwise CPU.

#### **3. Evaluation Process**
- **Validation Loss Calculation**: The trained model is tested on the validation dataset.
- **Ensemble Prediction**: Predictions from multiple models are averaged.
- **Performance Metrics**: Loss values are logged and included in the experiment report.

---

## **3. Results**
After each experiment, the following information is logged in the **experiment report**:
1. **Training Time**: Time taken for each training configuration.
2. **Validation Loss**: Final loss on the validation dataset.

### **3.1 Sample Experiment Report Output**
