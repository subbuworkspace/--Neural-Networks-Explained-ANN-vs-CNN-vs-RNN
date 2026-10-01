# 🧠 Neural Networks Explained: ANN vs CNN vs RNN

A beginner-friendly Python project to understand the basic concepts and differences between **ANN, CNN and RNN** using simple real-world examples.

The goal of this project is not just to train models, but to understand **why different neural networks are designed for different types of data**.

---

## 🚀 What Are ANN, CNN and RNN?

### 1️⃣ ANN — Artificial Neural Network

**Best suited for:** Structured / tabular data

ANN is the basic form of a neural network.

It consists mainly of:

```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

Example:

```text
Customer Age
Income
Credit Score
Loan Amount
      ↓
     ANN
      ↓
Prediction
```

ANN is commonly used for:

* Classification
* Regression
* Customer prediction
* Risk prediction
* Tabular business data

In this project, ANN is used to classify **Iris flowers**.

---

# 2️⃣ CNN — Convolutional Neural Network

**Best suited for:** Images and spatial data

CNN is designed to identify patterns in images.

Instead of looking at the entire image at once, CNN uses **filters/kernels** to detect useful features.

```text
Image
  ↓
Convolution
  ↓
Feature Maps
  ↓
Pooling
  ↓
More Features
  ↓
Flatten
  ↓
Dense Layer
  ↓
Prediction
```

For example, a CNN can gradually learn:

```text
Edges
 ↓
Shapes
 ↓
Parts
 ↓
Objects
```

CNN is commonly used for:

* Image classification
* Face detection
* Object detection
* Medical image analysis
* Computer vision

In this project, CNN is used to classify **MNIST handwritten digits**.

---

# 3️⃣ RNN — Recurrent Neural Network

**Best suited for:** Sequential data

RNN is designed to process data where the **order of information matters**.

For example:

```text
I → really → enjoyed → this → movie
```

The meaning of later words can depend on earlier words.

A simple RNN maintains information from previous steps:

```text
Input 1 → Hidden State
              ↓
Input 2 → Hidden State
              ↓
Input 3 → Hidden State
              ↓
Input 4 → Output
```

RNNs are commonly used for:

* Text classification
* Sentiment analysis
* Time-series data
* Speech-related tasks
* Sequence prediction

In this project, RNN is used for **IMDB movie-review sentiment classification**.

---

# 🔍 ANN vs CNN vs RNN

| Feature           | ANN                       | CNN                          | RNN                      |
| ----------------- | ------------------------- | ---------------------------- | ------------------------ |
| Full Name         | Artificial Neural Network | Convolutional Neural Network | Recurrent Neural Network |
| Best For          | Tabular data              | Images                       | Sequential data          |
| Main Idea         | Fully connected layers    | Filters + feature extraction | Memory of previous steps |
| Input             | Feature vectors           | Images                       | Sequences                |
| Important Concept | Weights                   | Convolution                  | Hidden state             |
| Example           | Customer prediction       | Image classification         | Sentiment analysis       |
| Project Dataset   | Iris                      | MNIST                        | IMDB                     |

---

# 🧩 Simple Way to Remember

Think about the type of data first.

```text
              What is my data?
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
    Tabular       Image       Sequence
       │            │            │
       ↓            ↓            ↓
      ANN          CNN          RNN
```

### Easy memory trick:

**ANN → Numbers**

**CNN → Pictures**

**RNN → Sequences**

---

# 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Matplotlib

---

# 📂 Project Structure

```text
Neural-Networks-ANN-CNN-RNN/
│
├── ann_tabular.py
├── cnn_image.py
├── rnn_text.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Neural-Networks-ANN-CNN-RNN.git
```

Move into the project:

```bash
cd Neural-Networks-ANN-CNN-RNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Projects

### ANN

```bash
python ann_tabular.py
```

The model learns to classify Iris flowers into:

```text
Setosa
Versicolor
Virginica
```

---

### CNN

```bash
python cnn_image.py
```

The CNN learns to classify handwritten digits:

```text
0 → 9
```

---

### RNN

```bash
python rnn_text.py
```

The RNN predicts whether a movie review is:

```text
Positive
or
Negative
```

---

# 🧠 What I Learned

Through this project, I explored:

* How neural networks learn from data
* Dense layers in ANN
* Activation functions
* Convolution and pooling in CNN
* Feature extraction from images
* Sequence processing in RNN
* Embeddings for text
* Model training and validation
* Classification using neural networks
* Why model architecture depends on the type of data

---

# 📌 ANN vs CNN vs RNN — In One Example

Imagine building an AI system for a hospital.

### ANN

Patient information:

```text
Age
Blood Pressure
Weight
Cholesterol
```

→ ANN can learn patterns from these structured features.

### CNN

Medical image:

```text
X-Ray / MRI / CT Scan
```

→ CNN can learn visual patterns.

### RNN

Patient history:

```text
Day 1 → Day 2 → Day 3 → Day 4
```

→ RNN can process information where sequence matters.

The important point is:

> **The neural network architecture should match the structure of the data.**

---

# 🎯 Project Goal

This is a learning project designed to make the difference between **ANN, CNN and RNN** easier to understand through Python implementation rather than only theory.

---

# 🚀 Future Improvements

Possible extensions:

* Add LSTM and GRU
* Compare RNN vs LSTM
* Add CNN image visualization
* Add confusion matrices
* Add Streamlit interface
* Add model performance comparison
* Add training/validation accuracy graphs
* Deploy the models as web applications

---

## 👨‍💻 Author

**Subrata Mondal**

Learning and building projects around:

**Python | Data Science | Machine Learning | Deep Learning | Generative AI | Computer Vision**

---

⭐ If this project helped you understand ANN, CNN and RNN, consider giving the repository a star!
