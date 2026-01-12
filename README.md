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
- ecommerce-review-summarizer/
- │
- ├── Add_and_Norm.py             
- ├── Cross_Attention.py           
- ├── CrossMultiHead.py           
- ├── Decoder.py                   
- ├── Encoder.py                  
- ├── FeedForward.py               
- ├── inputembeeding.py            
- ├── LinearAndSoftmax.py          
- ├── Masked_Multi_Head.py         
- ├── Masked_Single_Attention.py   
- ├── Multi_Head_Attention.py      
- ├── Positional_encoding.py      
- ├── Single_Head_Attention.py     
- ├── Transformer.py               
- ├── Vocublary_matrix.py         
- └── README.md

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
  - "Attention Is All You Need"
  - "The Illustrated Transformer - "Alammar J"



