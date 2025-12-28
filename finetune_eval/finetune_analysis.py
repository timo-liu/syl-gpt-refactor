import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from Definitions.Tokenizer import Tokenizer, TokenizerConfig
import json

compiler = {}
meta_compiler = {}

RUNS = {
	"finetune": "finetune_eval",
	"first_time": "finetune_eval/first_time_run",
}

CONFIGS_PATH = "C:/Users/timoy/OneDrive/Desktop/syl-gpt-refactor/Configs"

tokenizer_index = {}

# =========================
# LOAD DATA
# =========================

for run, folder in RUNS.items():
	for file in os.listdir(folder):
		if not file.endswith(".txt"):
			continue

		# ---------- NON-META FILES ----------
		if "meta" not in file:
			task, language, paradigm, cvc = file.rstrip(".txt").split("_")

			if f"{language}_{paradigm}" not in tokenizer_index:
				t_config = TokenizerConfig.load(
					os.path.join(CONFIGS_PATH, f"trained_{language}_{paradigm}_config.json")
				)
				tokenizer_index[f"{language}_{paradigm}"] = Tokenizer(t_config)

			with open(os.path.join(folder, file), "r") as f:
				prop, hits, total = [float(x) for x in f.readline().strip().split(",")[-3:]]

			compiler.setdefault(task, {})
			compiler[task].setdefault(language, {})
			compiler[task][language].setdefault(paradigm, {})
			compiler[task][language][paradigm].setdefault(run, [])

			compiler[task][language][paradigm][run].append({
				"cvc": cvc,
				"hits": hits,
				"totals": total
			})

		# ---------- META FILES (FINETUNE ONLY) ----------
		elif "meta" in file and run == "finetune":
			task, language, paradigm, cvc, _ = file.rstrip(".txt").split("_")

			if paradigm != "syl":
				continue

			if f"{language}_{paradigm}" not in tokenizer_index:
				t_config = TokenizerConfig.load(
					os.path.join(CONFIGS_PATH, f"trained_{language}_{paradigm}_config.json")
				)
				tokenizer_index[f"{language}_{paradigm}"] = Tokenizer(t_config)

			tokenizer = tokenizer_index[f"{language}_{paradigm}"]

			total_pretokens = 0
			total_fallback = 0
			unit_counts = []  # syllables OR words, depending on task

			with open(os.path.join(folder, file), "r") as f:
				for line in f:
					d = json.loads(line)

					reconstructed = tokenizer.decode(d["input"][:-1])
					deconstructed = tokenizer.tokenize(reconstructed, debug=True)

					total_pretokens += len(deconstructed)
					total_fallback += sum(x[1] for x in deconstructed)

					# ---- extract unit count (syllable or word) ----
					last_token = d["target"]
					count_str = tokenizer.decode([last_token]).strip("<").strip(">")
					try:
						unit_counts.append(int(count_str))
					except ValueError:
						pass

			meta_compiler.setdefault(task, {})
			meta_compiler[task].setdefault(language, {})
			meta_compiler[task][language].setdefault("syl", [])
			meta_compiler[task][language]["syl"].append({
				"cvc": cvc,
				"total_tokens": total_pretokens,
				"total_fallback": total_fallback,
				"unit_counts": unit_counts
			})
# =========================
# PLOTS: HIT RATES (RUNS COMBINED)
# =========================

os.makedirs("graphs", exist_ok=True)

for task in compiler:
	for language in compiler[task]:
		paradigms = compiler[task][language].keys()

		means = []
		errors = []
		labels = []
		ns = []

		for paradigm in paradigms:
			all_data = []

			for run in compiler[task][language][paradigm]:
				all_data.extend(compiler[task][language][paradigm][run])

			props = [d["hits"] / d["totals"] for d in all_data if d["totals"] > 0]

			if not props:
				continue

			mean_prop = np.mean(props)

			if len(props) > 1:
				sem = stats.sem(props)
				error = sem if sem > 0 else 0
			else:
				error = 0

			means.append(mean_prop)
			errors.append(error)
			labels.append(paradigm)
			ns.append(sum(d["totals"] for d in all_data))

		if len(means) < 2:
			continue

		plt.figure(figsize=(8, 5))
		bars = plt.bar(labels, means, yerr=errors, capsize=5)
		plt.ylabel("Proportion of Hits")
		plt.title(f"{task} | {language}")
		plt.ylim(0, 0.2)

		for bar, n in zip(bars, ns):
			height = bar.get_height()
			plt.text(
				bar.get_x() + bar.get_width() / 2,
				height + 0.02,
				f"n={int(n)}",
				ha="center",
				va="bottom"
			)

		plt.savefig(
			f"graphs/{task}_{language}_paradigm_comparison.png",
			bbox_inches="tight"
		)
		plt.close()

# =========================
# PLOTS: SYL FALLBACK (ENG VS SPAN)
# =========================

for task in meta_compiler:
	languages = []
	means = []
	errors = []
	token_counts = []

	for language in ("eng", "span"):
		if language not in meta_compiler[task]:
			continue
		if "syl" not in meta_compiler[task][language]:
			continue

		data = meta_compiler[task][language]["syl"]

		ratios = [
			d["total_fallback"] / d["total_tokens"]
			for d in data
			if d["total_tokens"] > 0
		]

		if not ratios:
			continue

		mean_ratio = np.mean(ratios)

		if len(ratios) > 1:
			sem = stats.sem(ratios)
			if sem > 0:
				conf = stats.t.interval(
					0.95,
					len(ratios) - 1,
					loc=mean_ratio,
					scale=sem
				)
				error = (conf[1] - conf[0]) / 2
			else:
				error = 0
		else:
			error = 0

		languages.append(language)
		means.append(mean_ratio)
		errors.append(error)
		token_counts.append(sum(d["total_tokens"] for d in data))

	if len(languages) < 2:
		continue

	plt.figure(figsize=(6, 5))
	bars = plt.bar(languages, means, yerr=errors, capsize=5)
	plt.ylabel("Average Byte Fallback Rate")
	plt.title(f"Syllable Fallback | {task}")
	plt.ylim(0, 1)

	for bar, tokens in zip(bars, token_counts):
		height = bar.get_height()
		plt.text(
			bar.get_x() + bar.get_width() / 2,
			height + 0.02,
			f"tokens={int(tokens)}",
			ha="center",
			va="bottom"
		)

	plt.savefig(
		f"graphs/{task}_syl_fallback_eng_vs_span.png",
		bbox_inches="tight"
	)
	plt.close()

# =========================
# PLOTS: SYLLABLE COUNT DISTRIBUTIONS
# =========================

for task in ("syllables", "word"):
	if task not in meta_compiler:
		continue

	for language in ("eng", "span"):
		if language not in meta_compiler[task]:
			continue
		if "syl" not in meta_compiler[task][language]:
			continue

		all_counts = []

		for d in meta_compiler[task][language]["syl"]:
			all_counts.extend(d["unit_counts"])

		if not all_counts:
			continue

		plt.figure(figsize=(7, 5))
		plt.hist(
			all_counts,
			bins=range(1, max(all_counts) + 2),
			align="left",
			edgecolor="black"
		)

		xlabel = "Syllable Count" if task == "syllables" else "Word Count"

		plt.xlabel(xlabel)
		plt.ylabel("Frequency")
		plt.title(f"{xlabel} Distribution | {language}")

		plt.savefig(
			f"graphs/{task}_{language}_count_distribution.png",
			bbox_inches="tight"
		)
		plt.close()