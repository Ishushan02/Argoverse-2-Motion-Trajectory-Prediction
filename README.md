# 🚗 Argoverse-2 Motion Trajectory Prediction 🧠📈

Welcome to the Argoverse 2 Motion Trajectory Prediction Challenge!  
This project focuses on building accurate **motion forecasting models** using the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html). You'll work with real-world driving data, pushing the limits of AI in **autonomous vehicle navigation**.

<div align="center">
  <img src="https://user-images.githubusercontent.com/placeholder/autonomous-car.png" alt="Autonomous Driving" width="600"/>
</div>

---

## 📌 About the Challenge

In real-world driving, predicting the motion of surrounding agents—vehicles, pedestrians, and cyclists—is **critical for safe navigation**.  
This dataset includes:

- 🕒 **11-second scenarios** (2s past, 9s future)
- 🧭 **Centroid + heading data** in **2D bird's-eye view**
- 🔄 **Sampled at 10Hz**
- 🚶‍♂️ Diverse agents with complex behaviors & interactions

### 🔍 Objective

Build ML models that predict **future trajectories** of agents with high accuracy in complex, crowded, and unpredictable scenes.

---

## 📁 Dataset Summary

| Feature         | Description                               |
|-----------------|-------------------------------------------|
| Sampling Rate   | 10Hz                                      |
| Input           | Past trajectories (50 Timestamps)         |
| Output          | Future trajectories (60 Timestamps)       |
| Agents          | Vehicles, pedestrians, cyclists           |
| Environment     | Real-world urban driving                  |

---

## 🔧 Model Architectures

I explored multiple architectures from basic LSTMs to advanced encoder-decoder frameworks and many other Architectures.

---

### **📘 Model A**

**Architecture 1**  
- 🧠 LSTM (1 unit) → MLP (3 layers)

**Architecture 2**  
- 🧱 Encoder (6-layer MLP) → LSTM → Decoder (6-layer MLP)

<div align="center">
  <img src="https://user-images.githubusercontent.com/placeholder/model-a.png" alt="Model A Diagram" width="600"/>
</div>

---

### **📙 Model B**

- 🔁 Simplified version of Model A
- 🔹 Encoder (3-layer MLP) → LSTM → Decoder (3-layer MLP)

---

### **📗 Model F**

- 🌀 Similar to Model A with **Skip Connections** added between MLP layers
- Better at handling information flow across the model

<div align="center">
  <img src="https://user-images.githubusercontent.com/placeholder/skip-connection.png" alt="Skip Connections" width="500"/>
</div>

---

### **📕 Model M (Final Model for Preliminary Results)**

- 🏁 Finalized architecture used in benchmarking
- Combines best practices from Models A, B, and F

---

## 🚀 Coming Soon

> I am actively working on implementing:
- ✅ Transformer-based architectures with attention mechanisms

---

## 💻 Get Started

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/Argoverse-2-Motion-Trajectory-Prediction.git
   cd Argoverse-2-Motion-Trajectory-Prediction
