import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Definitions.Tokenizer import Tokenizer, TokenizerConfig

# =========================
# CONFIG
# =========================

RUNS = {
	"finetune": "finetune_eval",
	"first_time": "finetune_eval/first_time_run",
}

CONFIGS_PATH = "C:/Users/timoy/OneDrive/Desktop/syl-gpt-refactor/Configs"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

tokenizer_index = {}
compiler = {}

# =========================
# LEVENSHTEIN
# =========================

def levenshtein(a, b):
	if a == b:
		return 0
	if len(a) == 0:
		return len(b)
	if len(b) == 0:
		return len(a)

	dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
	dp[:, 0] = range(len(a) + 1)
	dp[0, :] = range(len(b) + 1)

	for i in range(1, len(a) + 1):
		for j in range(1, len(b) + 1):
			cost = 0 if a[i - 1] == b[j - 1] else 1
			dp[i, j] = min(
				dp[i - 1, j] + 1,
				dp[i, j - 1] + 1,
				dp[i - 1, j - 1] + cost
			)
	return dp[-1, -1]

# =========================
# LOAD META G2P DATA ONLY
# =========================

for run, folder in RUNS.items():
	for file in os.listdir(folder):
		if not file.endswith(".txt"):
			continue
		if "meta" not in file:
			continue
		if "g2p" not in file:
			continue

		# expected: g2p_<language>_<paradigm>_<cvc>_meta.txt
		task, language, paradigm, cvc, _ = file.rstrip(".txt").split("_")

		tok_key = f"{language}_{paradigm}"
		if tok_key not in tokenizer_index:
			t_config = TokenizerConfig.load(
				os.path.join(CONFIGS_PATH, f"trained_{language}_{paradigm}_config.json")
			)
			tokenizer_index[tok_key] = Tokenizer(t_config)

		tokenizer = tokenizer_index[tok_key]

		distances = []

		with open(os.path.join(folder, file), "r") as f:
			for line in f:
				d = json.loads(line)

				target = tokenizer.decode(d["target"]).strip()
				generated = tokenizer.decode(d["generated"]).strip()
				if paradigm == "syl":
					print(f"{target} || {generated}")

				dist = levenshtein(generated, target)
				distances.append(dist)

		compiler.setdefault(language, {})
		compiler[language].setdefault(paradigm, {})
		compiler[language][paradigm].setdefault(run, [])
		compiler[language][paradigm][run].extend(distances)

# =========================
# PLOT: WITHIN-LANGUAGE PARADIGM COMPARISON
# =========================

for language in compiler:
	paradigms = []
	means = []
	errors = []
	ns = []

	for paradigm in compiler[language]:
		all_data = []

		for run in compiler[language][paradigm]:
			all_data.extend(compiler[language][paradigm][run])

		if not all_data:
			continue

		mean = np.mean(all_data)
		n = len(all_data)

		if n > 1:
			sem = stats.sem(all_data)
			error = sem if sem > 0 else 0
		else:
			error = 0

		paradigms.append(paradigm)
		means.append(mean)
		errors.append(error)
		ns.append(n)

	if len(paradigms) < 2:
		continue

	plt.figure(figsize=(7, 5))
	bars = plt.bar(paradigms, means, yerr=errors, capsize=5)
	plt.ylabel("Mean Levenshtein Distance")
	plt.title(f"G2P Meta Evaluation | {language}")

	for bar, n in zip(bars, ns):
		height = bar.get_height()
		plt.text(
			bar.get_x() + bar.get_width() / 2,
			height + 0.05,
			f"n={n}",
			ha="center",
			va="bottom"
		)

	plt.savefig(
		f"{GRAPH_DIR}/g2p_meta_edit_distance_{language}.png",
		bbox_inches="tight"
	)
	plt.close()
