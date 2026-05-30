# 🔮 LSTM Next-Word Predictor

A Streamlit web application that predicts the next word in a sentence using a trained LSTM (Long Short-Term Memory) neural network.

---

## 📁 Project Structure

```
next_words pred project/
├── app.py              # Main Streamlit application
├── lstm_model.h5       # Trained LSTM model
├── tokenizer.pkl       # Keras tokenizer (word ↔ index mappings)
├── max_lens.pkl        # Maximum sequence length used during training
└── README.md           # This file
```

---

## ⚙️ Requirements

- Python 3.8+
- The following Python packages:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `tensorflow` / `keras` | Load and run the LSTM model |
| `numpy` | Array operations |
| `h5py` | Patch model config for compatibility |
| `pickle` | Load tokenizer and max_lens |

---

## 🚀 Installation & Setup

### 1. Install dependencies

Open a terminal and run:

```bash
pip install streamlit tensorflow numpy h5py
```

### 2. Place all files in the same folder

Make sure `app.py`, `lstm_model.h5`, `tokenizer.pkl`, and `max_lens.pkl` are all in the same directory.

### 3. Run the app

Navigate to the project folder and launch Streamlit:

```bash
cd "path\to\next_words pred project"
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🖥️ How to Use

1. **Enter seed text** — Type a word, phrase, or sentence into the text area (e.g. `"the quick brown"`).
2. **Choose Top-N** — Use the slider to select how many word predictions to display (1–10).
3. **Click Predict** — Hit the `⟶ Predict` button.
4. **View results** — The app displays the top predicted next words ranked by probability, along with a preview of your sentence with the best prediction appended.

---

## 🧠 How It Works

```
User Input (text)
      │
      ▼
Tokenizer → converts words to integer sequences
      │
      ▼
Padding → sequence padded/truncated to max_len (748 tokens)
      │
      ▼
LSTM Model → outputs probability distribution over vocabulary (~8,200 words)
      │
      ▼
Top-N words → sorted by probability and displayed
```

### Model Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 748, 50) | 500,000 |
| LSTM | (None, 128) | 91,648 |
| Dense (softmax) | (None, 10000) | 1,290,000 |

- **Vocabulary size:** 8,213 words  
- **Sequence length:** 748 tokens  
- **Total parameters:** ~1.88 million

---

## 🛠️ Troubleshooting

### `File does not exist: app.py`
You are not in the correct directory. Run:
```bash
cd "path\to\your\project\folder"
streamlit run app.py
```

### `Failed to load model artifacts` (path error)
Ensure all four files (`app.py`, `lstm_model.h5`, `tokenizer.pkl`, `max_lens.pkl`) are in the **same folder**.

### `Unrecognized keyword argument: quantization_config`
This is a Keras version mismatch. The app automatically patches the `.h5` file on first run to fix this. If it persists, upgrade your packages:
```bash
pip install --upgrade tensorflow keras h5py
```

### Streamlit email prompt on first launch
Just press **Enter** to skip — no email is required.

---

## 📝 Notes

- The `.h5` model file is automatically patched on first run to remove incompatible config keys. This is a one-time operation and does not affect model weights or accuracy.
- Predictions are based on patterns learned during training. Results depend on the quality and domain of the training data.
- Words not in the training vocabulary will be treated as unknown and may affect prediction quality.

---

## 📄 License

This project is for educational and demonstration purposes.
