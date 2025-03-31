## 🧠 Sequence Labeling with RNN (PyTorch)

This notebook implements a simple **Recurrent Neural Network (RNN)** model for sequence labeling tasks such as **Named Entity Recognition (NER)** or **Part-of-Speech (POS)** tagging, using **PyTorch**.

---

### 🚀 Project Overview

This project is part of a deep learning course focused on **Natural Language Processing (NLP)**. The goal is to build a basic **sequence labeling model** using:

- **Word embeddings**
- **LSTM-based RNN**
- **Linear projection layer**
- **Cross-entropy loss for token classification**

---

### 🔧 Model Architecture

```
Input (word indices) ➡️ Embedding Layer ➡️ LSTM ➡️ Linear Layer ➡️ Output (logits for each tag)
```

- `nn.Embedding`: Maps input tokens to dense vectors
- `nn.LSTM`: Processes sequential data and learns temporal dependencies
- `nn.Linear`: Projects hidden state to output space (number of tags)
- `nn.CrossEntropyLoss`: Computes loss between predicted tag distribution and true labels

---

### 📦 Training Details

- Optimizer: `SGD` with learning rate `0.1`
- Loss function: `CrossEntropyLoss`
- Training loop includes:
  - Zeroing gradients
  - Forward pass
  - Reshaping output for loss
  - Backpropagation
  - Parameter update

---

### 📁 Files

- `SeqLabRNN`: PyTorch model class for sequence labeling
- Training loop for a fixed number of epochs (default: 5)
- Placeholder replaced with working implementation
- Fully self-contained for easy extension (e.g., batching, validation)

---

### 🛠️ How to Run

1. Install PyTorch
2. Load your `train_data` as a list of `(x, y)` pairs where:
   - `x`: token indices (`Tensor` of shape `[seq_len]`)
   - `y`: tag indices (`Tensor` of shape `[seq_len]`)
3. Run the notebook cells sequentially

---

### 🧪 Future Improvements

- Add batching for performance
- Add validation loop for monitoring
- Experiment with other optimizers (e.g., Adam)
- Add BiLSTM or CRF layers

---

### 📚 References

- Coursera: *Deep Learning for NLP*
- PyTorch Documentation

