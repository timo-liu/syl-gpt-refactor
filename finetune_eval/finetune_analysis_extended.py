# ============================================================
# ANALYSIS: bpe_uni_extended (single language, two paradigms)
# Mirrors hit-rate, guess-distribution, and meta (match-rate)
# (Fallback analysis removed)
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Definitions.Tokenizer import Tokenizer, TokenizerConfig

# -----------------------------
# CONFIGURATION
# -----------------------------
LANGUAGE = "eng"                     # single-language focus
PARADIGMS = {"bpe", "uni"}           # two-paradigm comparison
RUN_NAME = "bpe_uni_extended"
DATA_DIR = "finetune_eval/bpe_uni_extended"
CONFIGS_PATH = "C:/Users/timoy/OneDrive/Desktop/syl-gpt-refactor/Configs"
OUT_DIR = "graphs/bpe_uni_extended"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# DATA STRUCTURES
# -----------------------------
hit_compiler = {}     # task -> paradigm -> list[{hits, totals}]
meta_compiler = {}    # task -> paradigm -> meta dicts
guess_compiler = {}   # task -> paradigm -> guesses
tokenizer_index = {}

def add_significance_bracket(ax, x1, x2, y, h, text="*"):
	ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c="black")
	ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom")

# -----------------------------
# LOAD DATA
# -----------------------------
for file in os.listdir(DATA_DIR):
	if not file.endswith(".txt"):
		continue
	if "syllables" not in file and "word" not in file:
		continue

	parts = file.rstrip(".txt").split("_")
	if "meta" in file:
		task, language, paradigm, cvc, _ = parts
	else:
		task, language, paradigm, cvc = parts

	if language != LANGUAGE or paradigm not in PARADIGMS:
		continue

	tok_key = f"{language}_{paradigm}"
	if tok_key not in tokenizer_index:
		t_config = TokenizerConfig.load(
			os.path.join(CONFIGS_PATH, f"trained_{language}_{paradigm}_config.json")
		)
		tokenizer_index[tok_key] = Tokenizer(t_config)
	tokenizer = tokenizer_index[tok_key]

	# ---------- HIT RATE FILES ----------
	if "meta" not in file:
		with open(os.path.join(DATA_DIR, file), "r") as f:
			_, hits, total = [float(x) for x in f.readline().strip().split(",")[-3:]]

		hit_compiler.setdefault(task, {}).setdefault(paradigm, [])
		hit_compiler[task][paradigm].append({"hits": hits, "totals": total})

	# ---------- META FILES ----------
	else:
		total_pretokens = 0
		unit_counts = []
		guess_counts = []
		match_count = 0

		with open(os.path.join(DATA_DIR, file), "r") as f:
			for line in f:
				d = json.loads(line)

				reconstructed = tokenizer.decode(d["input"][:-1])
				deconstructed = tokenizer.tokenize(reconstructed, debug=False)
				total_pretokens += len(deconstructed)

				# gold
				gold = None
				try:
					gold = int(tokenizer.decode([d["target"]]).strip("<>"))
					unit_counts.append(gold)
				except ValueError:
					pass

				# guess
				try:
					guess = int(tokenizer.decode([d["prediction"]]).strip("<>"))
					guess_counts.append(guess)
				except ValueError:
					pass

				# match gold == input_tokens - 2
				if gold is not None:
					if gold == (len(d["input"]) - 2):
						match_count += 1

		meta_compiler.setdefault(task, {}).setdefault(paradigm, [])
		meta_compiler[task][paradigm].append({
			"total_tokens": total_pretokens,
			"unit_counts": unit_counts,
			"matches_input_minus_two": match_count
		})

		guess_compiler.setdefault(task, {}).setdefault(paradigm, [])
		guess_compiler[task][paradigm].extend(guess_counts)

# -----------------------------
# PLOT: HIT RATES (BPE vs UNI)
# -----------------------------
for task in hit_compiler:
	labels, means, errors, ns = [], [], [], []

	for paradigm in sorted(hit_compiler[task]):
		data = hit_compiler[task][paradigm]
		props = [d["hits"] / d["totals"] for d in data if d["totals"] > 0]
		if not props:
			continue

		labels.append(paradigm)
		means.append(np.mean(props))
		errors.append(stats.sem(props) if len(props) > 1 else 0)
		ns.append(sum(d["totals"] for d in data))

	if len(labels) != 2:
		continue

	fig, ax = plt.subplots(figsize=(6, 5))
	bars = ax.bar(labels, means, yerr=errors, capsize=5)

	y_max = max(m + e for m, e in zip(means, errors))
	if (means[0] + errors[0]) < (means[1] - errors[1]) or \
	   (means[1] + errors[1]) < (means[0] - errors[0]):
		add_significance_bracket(ax, 0, 1, y_max + 0.01, 0.01)

	ax.set_ylabel("Proportion of Hits")
	ax.set_title(f"{task} | {LANGUAGE} | BPE vs UNI")
	ax.set_ylim(0, 0.2)

	for bar, n in zip(bars, ns):
		ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
				f"n={int(n)}", ha="center")

	plt.savefig(f"{OUT_DIR}/{task}_{LANGUAGE}_bpe_vs_uni_hits.png",
				bbox_inches="tight")
	plt.close()

# -----------------------------
# PLOT: GUESS DISTRIBUTIONS
# -----------------------------
for task in guess_compiler:
	for paradigm in guess_compiler[task]:
		data = guess_compiler[task][paradigm]
		if not data:
			continue

		plt.figure(figsize=(6, 5))
		plt.hist(data, bins=range(1, max(data) + 2),
				 align="left", edgecolor="black")
		xlabel = "Guessed Syllable Count" if task == "syllables" else "Guessed Word Count"
		plt.xlabel(xlabel)
		plt.ylabel("Frequency")
		plt.title(f"Guess Distribution | {task} | {LANGUAGE} | {paradigm}")

		plt.savefig(f"{OUT_DIR}/guesses_{task}_{LANGUAGE}_{paradigm}.png",
					bbox_inches="tight")
		plt.close()

# -----------------------------
# PLOT: MATCH RATE (gold == input-2)
# -----------------------------
for task in meta_compiler:
	labels, rates = [], []

	for paradigm in sorted(meta_compiler[task]):
		all_data = meta_compiler[task][paradigm]
		total_matches = sum(d["matches_input_minus_two"] for d in all_data)
		total = sum(len(d["unit_counts"]) for d in all_data)
		if total == 0:
			continue

		labels.append(paradigm)
		rates.append(total_matches / total)

	if len(labels) != 2:
		continue

	plt.figure(figsize=(6, 5))
	bars = plt.bar(labels, rates, edgecolor="black")
	plt.ylabel("Proportion Gold == Input Tokens - 2")
	plt.title(f"Match Rate | {task} | {LANGUAGE}")
	plt.ylim(0, 1)

	for bar, r in zip(bars, rates):
		plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
				 f"{r:.2f}", ha="center")

	plt.savefig(f"{OUT_DIR}/match_rate_{task}_{LANGUAGE}_bpe_vs_uni.png",
				bbox_inches="tight")
	plt.close()
