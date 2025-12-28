import argparse
from Definitions.Model import *
from torch import nn
import torch
import torch.nn.functional as F
import os
import numpy as np
import json
import math

parser = argparse.ArgumentParser()
parser.add_argument("finetuned_weight", type=str, help="Path to the weight that needs to be evaluated")
parser.add_argument("language", type=str, help="hee hee")
parser.add_argument("paradigm", type=str, help="hoo hoo")
parser.add_argument("cvc", type=int, help="what cross val counter to pull")
parser.add_argument("task", type=str, help="Word? Syllable? g2p?")
parser.add_argument("bins_path", type=str, help="bins/")

cli_args = parser.parse_args()

device = "cuda"
config = GPTConfig()
model = GPT(config)
state_dict = torch.load(cli_args.finetuned_weight, map_location="cpu")["model"]
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()})

for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()

model = model.to(device)
model.eval()

pattern = os.path.join(cli_args.bins_path, f"{cli_args.task}_{cli_args.language}_{cli_args.paradigm}_test_{cli_args.cvc}_000000.bin")

with open(pattern, "rb") as f:
    header = np.frombuffer(f.read(256*4), dtype=np.int32)
    tokens = np.frombuffer(f.read(), dtype=np.uint16)

out_dir = "finetune_eval"
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(
    out_dir,
    f"{cli_args.task}_{cli_args.language}_{cli_args.paradigm}_{cli_args.cvc}.txt"
)

meta_file = os.path.join(
    out_dir,
    f"{cli_args.task}_{cli_args.language}_{cli_args.paradigm}_{cli_args.cvc}_meta.txt"
)

pad_to_multiple = 16  # common padding factor

if cli_args.task in ["syllables", "word"]:
    separators = (tokens > (257 if cli_args.paradigm == "syl" else 259)) & (
                 tokens <= (285 if cli_args.paradigm == "syl" else 287))
    sep_idx = np.where(separators)[0]
    splits = np.split(tokens, sep_idx + 1)

    hits = 0
    tot = 0

    with open(meta_file, "w") as meta:
        for s in splits:
            if len(s) <= 3:
                continue
            meta_json = {}
            inp = s[1:-1]
            target = int(s[-1])
            meta_json["input"] = inp.tolist()
            meta_json["token_count"] = len(inp)
            meta_json["target"] = target
            T_unpadded = len(inp)
            pad_to = ((T_unpadded + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
            pad_needed = pad_to - T_unpadded

            tokens_tensor = torch.tensor(inp, dtype=torch.long, device=device)
            tokens_tensor = F.pad(tokens_tensor, (0, pad_needed), value=0)

            with torch.inference_mode():
                logits = model(tokens_tensor, inference=True)
                last_token_logits = logits[0, T_unpadded - 1, :]

                if cli_args.paradigm == "syl":
                    sep_min, sep_max = 258, 285
                else:
                    sep_min, sep_max = 260, 287
                vocab_size = last_token_logits.size(0)

                separator_mask = (
                    (torch.arange(vocab_size, device=last_token_logits.device) >= sep_min) &
                    (torch.arange(vocab_size, device=last_token_logits.device) <= sep_max)
                )
                masked_logits = last_token_logits.masked_fill(~separator_mask, float('-inf'))
                pred = torch.argmax(masked_logits).item()
                meta_json["prediction"] = pred

            if pred == target:
                hits += 1
            tot += 1
            meta.write(json.dumps(meta_json) + "\n")

    acc = hits / tot if tot > 0 else 0.0
    line = f"{cli_args.task},{cli_args.language},{cli_args.paradigm},{cli_args.cvc},{acc:.6f},{hits},{tot}\n"
    print(f"Accuracy: {acc:.4f} ({hits}/{tot})")
    with open(out_file, "a") as f:
        f.write(line)

else:  # G2P task
    eos_id = 286 if cli_args.paradigm == "syl" else 288
    splits = np.split(tokens, np.where(tokens == eos_id)[0])

    hits = 0
    tot = 0

    with open(meta_file, "w") as meta:
        for s in splits:
            split_id = 256 if cli_args.paradigm == "syl" else 258
            split_idx = np.where(s == split_id)[0]
            if len(split_idx) == 0:
                continue
            split_idx = split_idx[0]

            inp = s[:split_idx]
            actual = s[split_idx + 1:]

            # --- Generation ---
            generated = []
            prompt_tokens = inp.tolist()

            with torch.inference_mode():
                for _ in range(25):

                    # --- 2. Prepare Inputs (The "Hard Way") ---

                    # Current unpadded length
                    T_unpadded = len(prompt_tokens)

                    # Pad to next multiple
                    pad_to = math.ceil(T_unpadded / pad_to_multiple) * pad_to_multiple
                    assert pad_to % pad_to_multiple == 0

                    pad_needed = pad_to - T_unpadded

                    # Create padded token tensor (no carry-over)
                    tokens_tensor = torch.tensor(
                        prompt_tokens, dtype=torch.long, device=device
                    )
                    tokens_tensor = F.pad(tokens_tensor, (0, pad_needed), value=0)

                    T_padded = len(tokens_tensor)

                    logits = model(tokens_tensor, inference=True)

                    # --- 4. Get the Next Token ---
                    last_token_logits = logits[0, T_unpadded - 1, :]

                    # --- 5. Argmax (greedy) ---
                    next_token_id = torch.argmax(last_token_logits).item()

                    generated.append(next_token_id)

                    # --- 6. Append and Loop ---
                    if next_token_id == eos_id:
                        break

                    prompt_tokens.append(next_token_id)
                
            # --- Evaluation ---
            actual_trimmed = [int(x) for x in actual if x != eos_id]
            gen_trimmed = [int(x) for x in generated if x != eos_id]
            correct = gen_trimmed == actual_trimmed

            if correct:
                hits += 1
            tot += 1

            meta_json = {
                "input": inp.tolist(),
                "target": actual_trimmed,
                "generated": gen_trimmed,
                "correct": correct
            }
            meta.write(json.dumps(meta_json) + "\n")

    acc = hits / tot if tot > 0 else 0.0
    line = f"{cli_args.task},{cli_args.language},{cli_args.paradigm},{cli_args.cvc},{acc:.6f},{hits},{tot}\n"
    print(f"Accuracy: {acc:.4f} ({hits}/{tot})")
    with open(out_file, "a") as f:
        f.write(line)