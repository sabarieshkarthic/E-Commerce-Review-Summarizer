# 🛒 E-Commerce Review Summarizer (Transformer)

This project implements a **Transformer-based sequence-to-sequence model** for **summarizing e-commerce product reviews**, built **entirely from scratch using Python and NumPy**.

The implementation includes custom encoder–decoder blocks, attention mechanisms, embeddings, loss computation, backpropagation, and inference logic without relying on high-level deep learning frameworks.

---

## 📌 Project Overview

The system converts long product reviews into short summaries using a **Transformer encoder–decoder architecture**.

The pipeline consists of:
- Vocabulary and embedding construction
- Encoder stack with self-attention
- Decoder stack with masked self-attention and cross-attention

---

## 📂 Project Structure
ecommerce-review-summarizer/
│
├── Add_and_Norm.py              # Residual connection + layer normalization
├── Cross_Attention.py           # Encoder–decoder attention
├── CrossMultiHead.py            # Multi-head cross-attention
├── Decoder.py                   # Transformer decoder block
├── Encoder.py                   # Transformer encoder block
├── FeedForward.py               # Position-wise feedforward network
├── inputembeeding.py            # Token + positional embeddings
├── LinearAndSoftmax.py          # Output projection and softmax
├── Masked_Multi_Head.py         # Masked multi-head attention
├── Masked_Single_Attention.py   # Masked single-head attention
├── Multi_Head_Attention.py      # Multi-head self-attention
├── Positional_encoding.py       # Positional encoding
├── Single_Head_Attention.py     # Single-head attention
├── Transformer.py               # Full Transformer model
├── Vocublary_matrix.py          # Vocabulary and shared embedding matrix
└── README.md

---

## 🧠 Model Architecture

The model follows a **standard Transformer encoder–decoder design**.

### Encoder
- Input embedding + positional encoding  
- Multi-head self-attention  
- Feedforward network  
- Residual connections and layer normalization  

### Decoder
- Masked self-attention  
- Cross-attention with encoder outputs  
- Feedforward network  
- Residual connections and layer normalization  

### Output
- Linear projection  
- Softmax over vocabulary  

---

## 🔍 Text Preprocessing

- Lowercasing  
- Tokenization by whitespace  
- Vocabulary indexing  
- Special tokens:
  - `<start>`
  - `<end>`
  - `<pad>`
  - `<unk>`
- Sequence padding and truncation  
- Shared vocabulary for encoder and decoder  

---

## 🔁 Training Pipeline

- Sequence-to-sequence training   
- Token-level cross-entropy loss  
- Manual backpropagation  
- Parameter updates using **gradient descent**  

---

## ⚙️ Training Configuration

| Parameter | Value |
|---------|------|
| Encoder blocks | 1 |
| Decoder blocks | 1 |
| Optimizer | Gradient Descent |
| Learning rate | 0.01 |
| Epochs | Up to 2000 |
| Loss function | Cross-Entropy |

---
## References:
  - title: "Attention Is All You Need"
  - title: "The Illustrated Transformer - "Alammar J"



