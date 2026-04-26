import sys
import os
import pickle
import torch
import torch.nn as nn
import numpy as np

SAVE_DIR = "saved_models"   # folder where best_model.pt is saved
MAX_LEN = 53

class EEGtoTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        
        self.word_encoder = nn.Sequential(
            nn.Linear(105 * 6, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 53, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        B, T, N, F = x.shape
        
        x = x.reshape(B, T, N * F)
        x = self.word_encoder(x)
        x = x + self.pos_embedding[:, :T, :]
        x = self.transformer(x)
        
        return self.fc_out(x)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    with open(os.path.join(SAVE_DIR, "model_config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    # Load vocab
    with open(os.path.join(SAVE_DIR, "word2idx.pkl"), "rb") as f:
        word2idx = pickle.load(f)

    with open(os.path.join(SAVE_DIR, "idx2word.pkl"), "rb") as f:
        idx2word = pickle.load(f)
    
    model = EEGtoTextModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"]
    ).to(device)
    
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location=device)
    )
    
    model.eval()
    
    return model, word2idx, idx2word, device


def preprocess_sample(sample_data, word2idx):
    eegs = sample_data["eeg_vectors"]
    words = sample_data["words"]
    
    X = []
    
    for eeg, word in zip(eegs, words):
        eeg = np.array(eeg)
        
        if len(eeg) != 630:
            continue
        
        if word in word2idx:
            eeg = eeg.reshape(6, 105).T  # (105,6)
            X.append(eeg)
    
    # pad
    while len(X) < MAX_LEN:
        X.append(np.zeros((105,6)))
    
    X = X[:MAX_LEN]
    
    return np.array(X)

def predict(model, eeg_sentence, idx2word, device):
    with torch.no_grad():
        x = torch.tensor(eeg_sentence, dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(x)
        preds = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()
    
    words = []
    for idx in preds:
        if idx in idx2word:
            words.append(idx2word[idx])
    
    return " ".join(words)


def main():
    
    sample_path = r"test_samples\sample_3.pkl"
    
    if not os.path.exists(sample_path):
        print("File not found:", sample_path)
        sys.exit(1)
    
    # Load model
    model, word2idx, idx2word, device = load_model()
    
    # Load sample
    with open(sample_path, "rb") as f:
        sample_data = pickle.load(f)
    
    # Preprocess
    eeg_sentence = preprocess_sample(sample_data, word2idx)
    
    # Predict
    prediction = predict(model, eeg_sentence, idx2word, device)
    
    print("\n Predicted Sentence:\n")
    print(prediction)

def run_inference(sample_path):
    model, word2idx, idx2word, device = load_model()
    
    with open(sample_path, "rb") as f:
        sample_data = pickle.load(f)
    
    eeg_sentence = preprocess_sample(sample_data, word2idx)
    prediction = predict(model, eeg_sentence, idx2word, device)
    
    return prediction


if __name__ == "__main__":
    main()
