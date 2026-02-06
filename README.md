# Electrical Fault Detection & Classification using Deep Learning

### 📌 Project Overview

This repository contains a specialized Deep Learning pipeline designed to detect and classify faults in electrical power systems. Using 3-phase current and voltage signals, the model can identify six different system states (Healthy, LG, LL, LLG, LLL, LLLG) with near-instantaneous inference.

By leveraging **PyTorch** and **Physics-Informed Data Engineering**, this project demonstrates how AI can replace or augment traditional fixed-threshold protection relays to create more resilient smart grids.

### ⚡ The Problem

Traditional protection relays often struggle with:

* **Nuance Tripping:** Mistaking transient noise for a fault.
* **Complex Discrimination:** Difficulty distinguishing between different types of multi-phase faults.
* **High Impedance:** Subtle signal changes that don't trigger simple overcurrent thresholds.

### 🛠️ Technical Architecture

#### 1. Data Engineering & Preprocessing

The "garbage in, garbage out" rule is critical in power systems. This project implements a rigorous preprocessing pipeline:

* **Feature Set:** Analysis of a 6-dimensional input space ( and ).
* **Normalization:** Implementation of `StandardScaler` to handle the high variance between fault currents () and system voltages (), ensuring gradient stability during backpropagation.
* **Ground-Truth Logic:** Incorporation of binary phase/ground indicators to assist the model in learning the mathematical relationship between grounded and ungrounded faults.

#### 2. Neural Network Implementation

The classifier is built using a Multi-Layer Perceptron (MLP) architecture in **PyTorch**:

* **Input Layer:** 6 features (3-phase I/V).
* **Hidden Layers:** Fully connected layers with ReLU activation, optimized for high-speed classification.
* **Output Layer:** 6-way Softmax classification for specific fault types.
* **Hardware Acceleration:** Full support for **CUDA**-enabled GPUs for rapid training and real-time inference simulation.

### 📊 Performance & Results

The model achieves **100% accuracy** on the test dataset, effectively mapping every fault signature to its correct physical category.

| Fault Type | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| no fault | 1.00 | 1.00 | 1.00 |
| LG / LLG | 1.00 | 1.00 | 1.00 |
| LLL / LLLG | 1.00 | 1.00 | 1.00 |

### 🚀 Getting Started

1. **Clone the repo:**
```bash
git clone https://github.com/omar-ehab04/electrical-fault-detection.git

```


2. **Install Dependencies:**
```bash
pip install torch pandas numpy scikit-learn matplotlib

```


3. **Run the Notebooks:**
* `electrical_fault_detection(1).ipynb`: Data Exploration and Preprocessing.
* `electrical_fault_detection(2).ipynb`: Model Training and Evaluation.



---
