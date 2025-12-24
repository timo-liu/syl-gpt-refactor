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
  separators = (tokens > (257 if cli_args.paradigm == "syl" else 259)) & (tokens <= (285 if cli_args.paradigm == "syl" else 287))
  sep_idx = np.where(separators)[0]
  splits = np.split(tokens, sep_idx + 1)

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

hits = 0
tot = 0

pad_to_multiple = 16

meta = open(meta_file, "w")

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

meta.close()
acc = hits / tot if tot > 0 else 0.0
line = f"{cli_args.task},{cli_args.language},{cli_args.paradigm},{cli_args.cvc},{acc:.6f},{hits},{tot}\n"

print(f"Accuracy: {acc:.4f} ({hits}/{tot})")

with open(out_file, "a") as f:
	f.write(line)