import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import argparse
import json
from transformers import AutoModel

# --- Parse CLI args (minimal) ---
parser = argparse.ArgumentParser()
parser.add_argument("language", type=str)
parser.add_argument("task", type=str)
parser.add_argument("cross_val_counter", type=int)
parser.add_argument("bins_path", type=str)
cli_args = parser.parse_args()

# --- Setup device and DDP ---
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master_process = (ddp_rank == 0)

# --- Hyperparameters ---
B = 8  # batch size per device
T = 64 # sequence length
num_iterations = 256  # example; tune as needed
val_loss_every = 16

# --- Load GPT-2 model ---
model = AutoModel.from_pretrained("gpt2")
model = model.to(device).bfloat16()
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

# --- Dummy optimizer for example ---
optimizer = torch.optim.Adam(raw_model.parameters(), lr=1e-4)

# --- Dummy data loader (implement your own based on bins) ---
def load_tokens_for_training(path_pattern):
    # This should yield batches (x, y) shaped (B, T)
    # For simplicity, here we just load one file and slice tokens into batches
    # You want to implement a proper loader reading from multiple bin files with sharding for DDP
    bin_file = os.path.join(cli_args.bins_path, f"{cli_args.task}_{cli_args.language}_train_{cli_args.cross_val_counter}_000000.bin")
    with open(bin_file, "rb") as f:
        _ = np.frombuffer(f.read(256*4), dtype=np.int32)  # skip header
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    # Remove separator tokens etc as needed, then slice into batches of BxT
    # For demo, let's just split tokens into chunks of size T, batchify B samples at a time
    sequences = []
    for i in range(0, len(tokens) - T - 1, T):
        sequences.append(tokens[i:i+T+1])  # input + target
    # Batchify
    for i in range(0, len(sequences) - B + 1, B):
        batch = sequences[i:i+B]
        x = np.stack([seq[:-1] for seq in batch])
        y = np.stack([seq[1:] for seq in batch])
        yield torch.tensor(x, device=device, dtype=torch.long), torch.tensor(y, device=device, dtype=torch.long)

train_data_iter = load_tokens_for_training(cli_args.bins_path)

# --- Evaluation function from your pseudocode ---
def evaluate_model(model, device, cli_args):
    model.eval()
    pad_to_multiple = 16
    hits, tot = 0, 0

    pattern = os.path.join(cli_args.bins_path, f"{cli_args.task}_{cli_args.language}_test_{cli_args.cross_val_counter}_000000.bin")
    with open(pattern, "rb") as f:
        _ = np.frombuffer(f.read(256*4), dtype=np.int32)  # skip header
        tokens = np.frombuffer(f.read(), dtype=np.uint16)

    out_dir = "finetune_eval"
    os.makedirs(out_dir, exist_ok=True)
    meta_file = os.path.join(out_dir, f"{cli_args.task}_{cli_args.language}_eval_{cli_args.cross_val_counter}_meta.txt")

    with open(meta_file, "w") as meta:
        if cli_args.task in ["syllables", "word"]:
            separators = (tokens > 259) & (tokens <= 287)
            sep_idx = np.where(separators)[0]
            splits = np.split(tokens, sep_idx + 1)

            for s in splits:
                if len(s) <= 3:
                    continue
                inp = s[1:-1]
                target = int(s[-1])
                T_unpadded = len(inp)
                pad_to = ((T_unpadded + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
                pad_needed = pad_to - T_unpadded

                tokens_tensor = torch.tensor(inp, dtype=torch.long, device=device)
                tokens_tensor = F.pad(tokens_tensor, (0, pad_needed), value=0)

                with torch.inference_mode():
                    logits = model(tokens_tensor, inference=True)
                    last_token_logits = logits[0, T_unpadded - 1, :]

                    sep_min, sep_max = 260, 287
                    vocab_size = last_token_logits.size(0)
                    separator_mask = ((torch.arange(vocab_size, device=last_token_logits.device) >= sep_min) &
                                      (torch.arange(vocab_size, device=last_token_logits.device) <= sep_max))
                    masked_logits = last_token_logits.masked_fill(~separator_mask, float('-inf'))
                    pred = torch.argmax(masked_logits).item()

                correct = pred == target
                hits += int(correct)
                tot += 1

                meta.write(json.dumps({
                    "input": inp.tolist(),
                    "token_count": len(inp),
                    "target": target,
                    "prediction": pred,
                    "correct": correct
                }) + "\n")

        else:
            # Implement G2P evaluation here if needed
            pass

    accuracy = hits / tot if tot > 0 else 0.0
    if master_process:
        print(f"Evaluation accuracy: {accuracy:.4f} ({hits}/{tot})")
    model.train()
    return accuracy


# --- Training loop ---
for step in range(num_iterations):
    model.train()
    try:
        x, y = next(train_data_iter)
    except StopIteration:
        # Restart iterator
        train_data_iter = load_tokens_for_training(cli_args.bins_path)
        x, y = next(train_data_iter)

    optimizer.zero_grad()
    logits = model(x, inference=True)
    # GPT-2 logits shape might differ; assume logits is (B, T, vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()

    if step % 50 == 0 and master_process:
        print(f"Step {step}/{num_iterations} train loss: {loss.item():.4f}")

    if (step + 1) % val_loss_every == 0 and master_process:
        acc = evaluate_model(raw_model, device, cli_args)

if master_process:
    acc = evaluate_model(raw_model, device, cli_args)
    print(f"Final evaluation accuracy: {acc:.4f}")

dist.destroy_process_group()
