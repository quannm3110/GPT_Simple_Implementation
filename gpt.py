import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *
from model import GPTLanguageModel
torch.manual_seed(RANDOM_SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def import_data():
    """Import data

    Returns:
        text (str): text
        vocab_size (int): length of vocab
        encode (lambda function): lambda encode function
        decode (lambda function): lambda decode function
    """
    
    # Load data
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Create a mapping for encoding and decoding
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return text, vocab_size, encode, decode

def get_batch(data):
    """Prepare data batch

    Args:
        data (text): text data to randomize extract the batch from

    Returns:
        x (torch tensor): context/input tensor
        y (torch tensor): target tensor
    """
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split=='train' else val_data
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    
    # Import data 
    text, vocab_size, encode, decode = import_data()
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model
    model = GPTLanguageModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT)
    m = model.to(DEVICE)

    # Print number of parameters in model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):

        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ")

        # Sample a batch of data points
        xb, yb = get_batch(train_data)

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate fro model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__=='__main__':
    main()