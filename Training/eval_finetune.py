import argparse
from Definitions.Model import *
from torch import nn
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("finetuned_weight", type=str, help="Path to the weight that needs to be evaluated")
parser.add_argument("language", type=str, help="hee hee")
parser.add_argument("paradigm", type=str, help="hoo hoo")
parser.add_argument("cvc", type=int, help="what cross val counter to pull")
parser.add_argument("task", type=str, help="Word? Syllable?")
parser.add_argument("bins_path", type=str, help="bins/")

cli_args = parser.parse_args()

device = "cuda"
config = GPTConfig()
model = GPT(config)
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in torch.load(cli_args.finetuned_weight, map_location="cpu")["model"].items()})
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()

model = model.to(device)
model.eval()

pattern = os.path.join(cli_args.bins_path, f"{cli_args.task}_{cli_args.language}_{cli_args.paradigm}_test_{cli_args.cvc}_000000.bin")

with open(pattern, "rb") as f:
  header = np.frombuffer(f.read(256*4), dtype=np.int32)
  tokens = np.frombuffer(f.read(), dtype=np.uint16)
  separators = (tokens >= 258) & (tokens <= 287)
  sep_idx = np.where(separators)[0]
  splits = np.split(tokens, sep_idx + 1)

hits = 0
tot = 0

for s in splits:
	if len(s) < 2:
		continue

	inp = s[:-1]
	target = int(s[-1])

	tokens_tensor = torch.tensor(inp, dtype=torch.long, device=device).unsqueeze(0)
	T_unpadded = tokens_tensor.size(1)

	with torch.no_grad():
		logits = model(tokens_tensor, inference=True)
		last_token_logits = logits[0, T_unpadded - 1, :]
		pred = torch.argmax(last_token_logits).item()

	if pred == target:
		hits += 1

	tot += 1

acc = hits / tot if tot > 0 else 0.0
print(f"Accuracy: {acc:.4f} ({hits}/{tot})")