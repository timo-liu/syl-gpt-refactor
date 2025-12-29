import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Definitions.Tokenizer import Tokenizer, TokenizerConfig

# -----------------------------
# CONFIGURATION
# -----------------------------
RUNS = {
    "finetune": "finetune_eval",
    "first_time": "finetune_eval/first_time_run",
}

CONFIGS_PATH = "C:/Users/timoy/OneDrive/Desktop/syl-gpt-refactor/Configs"
os.makedirs("graphs", exist_ok=True)

# -----------------------------
# DATA STRUCTURES
# -----------------------------
compiler = {}       # Tracks hit rates
meta_compiler = {}  # Tracks meta info (tokens, matches)
guess_compiler = {} # Tracks guesses per paradigm
tokenizer_index = {}

# -----------------------------
# DATA LOADING
# -----------------------------
for run, folder in RUNS.items():
    for file in os.listdir(folder):
        if not file.endswith(".txt") or ("syllables" not in file and "word" not in file):
            continue

        parts = file.rstrip(".txt").split("_")
        if "meta" in file:
            task, language, paradigm, cvc, _ = parts
        else:
            task, language, paradigm, cvc = parts

        tok_key = f"{language}_{paradigm}"
        if tok_key not in tokenizer_index:
            t_config = TokenizerConfig.load(
                os.path.join(CONFIGS_PATH, f"trained_{language}_{paradigm}_config.json")
            )
            tokenizer_index[tok_key] = Tokenizer(t_config)
        tokenizer = tokenizer_index[tok_key]

        # ---------- NON-META FILES (hit rates) ----------
        if "meta" not in file:
            with open(os.path.join(folder, file), "r") as f:
                prop, hits, total = [float(x) for x in f.readline().strip().split(",")[-3:]]

            compiler.setdefault(task, {}).setdefault(language, {}).setdefault(paradigm, {}).setdefault(run, [])
            compiler[task][language][paradigm][run].append({"hits": hits, "totals": total})

        # ---------- META FILES ----------
        elif run == "finetune":
            total_pretokens = 0
            total_fallback = 0
            unit_counts = []
            guess_counts = []
            match_count = 0
            debug_mode = paradigm == "syl"

            with open(os.path.join(folder, file), "r") as f:
                for line in f:
                    d = json.loads(line)

                    reconstructed = tokenizer.decode(d["input"][:-1])
                    deconstructed = tokenizer.tokenize(reconstructed, debug=debug_mode)

                    if debug_mode:
                        total_pretokens += len(deconstructed)
                        total_fallback += sum(x[1] for x in deconstructed)

                    # ---- gold count ----
                    gold = None
                    try:
                        gold = int(tokenizer.decode([d["target"]]).strip("<>"))
                        unit_counts.append(gold)
                    except ValueError:
                        pass

                    # ---- guessed count ----
                    try:
                        guess = int(tokenizer.decode([d["prediction"]]).strip("<>"))
                        guess_counts.append(guess)
                    except ValueError:
                        pass

                    # ---- count matches for ALL paradigms ----
                    if gold is not None:
                        input_token_count = len(d["input"]) - 2
                        if gold == input_token_count:
                            match_count += 1

            # Store meta info per paradigm
            meta_compiler.setdefault(task, {}).setdefault(language, {}).setdefault(paradigm, [])
            meta_compiler[task][language][paradigm].append({
                "total_tokens": total_pretokens,
                "total_fallback": total_fallback,
                "unit_counts": unit_counts,
                "matches_input_minus_two": match_count
            })

            # Store guesses per paradigm
            guess_compiler.setdefault(task, {}).setdefault(language, {}).setdefault(paradigm, [])
            guess_compiler[task][language][paradigm].extend(guess_counts)

# -----------------------------
# PLOTS: HIT RATES
# -----------------------------
for task in compiler:
    for language in compiler[task]:
        labels, means, errors, ns = [], [], [], []

        for paradigm in compiler[task][language]:
            all_data = []
            for run in compiler[task][language][paradigm]:
                all_data.extend(compiler[task][language][paradigm][run])

            props = [d["hits"] / d["totals"] for d in all_data if d["totals"] > 0]
            if not props:
                continue

            labels.append(paradigm)
            means.append(np.mean(props))
            errors.append(stats.sem(props) if len(props) > 1 else 0)
            ns.append(sum(d["totals"] for d in all_data))

        if len(labels) < 2:
            continue

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, means, yerr=errors, capsize=5)
        plt.ylabel("Proportion of Hits")
        plt.title(f"{task} | {language}")
        plt.ylim(0, 0.2)

        for bar, n in zip(bars, ns):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={int(n)}", ha="center")

        plt.savefig(f"graphs/{task}_{language}_paradigm_comparison.png", bbox_inches="tight")
        plt.close()

# -----------------------------
# PLOTS: GUESS DISTRIBUTIONS
# -----------------------------
for task in guess_compiler:
    for language in guess_compiler[task]:
        for paradigm in guess_compiler[task][language]:
            data = guess_compiler[task][language][paradigm]
            if not data:
                continue

            plt.figure(figsize=(7, 5))
            plt.hist(data, bins=range(1, max(data) + 2), align="left", edgecolor="black")
            xlabel = "Guessed Syllable Count" if task == "syllables" else "Guessed Word Count"
            plt.xlabel(xlabel)
            plt.ylabel("Frequency")
            plt.title(f"Guess Distribution | {task} | {language} | {paradigm}")

            plt.savefig(f"graphs/guesses_{task}_{language}_{paradigm}.png", bbox_inches="tight")
            plt.close()

# -----------------------------
# PLOTS: MATCH RATE (gold == input-2) PER LANGUAGE
# -----------------------------
for task in meta_compiler:
    for language in meta_compiler[task]:
        labels, match_rates = [], []

        for paradigm in meta_compiler[task][language]:
            all_data = meta_compiler[task][language][paradigm]
            total_matches = sum(d.get("matches_input_minus_two", 0) for d in all_data)
            total_sentences = sum(len(d.get("unit_counts", [])) for d in all_data)

            if total_sentences == 0:
                continue

            labels.append(paradigm)
            match_rates.append(total_matches / total_sentences)

        if not labels:
            continue

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, match_rates, color="skyblue", edgecolor="black")
        plt.ylabel("Proportion of Gold == Input Tokens - 2")
        plt.title(f"Match Rate by Paradigm | {task} | {language}")
        plt.ylim(0, 1)

        for bar, rate in zip(bars, match_rates):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{rate:.2f}", ha="center")

        plt.savefig(f"graphs/match_rate_{task}_{language}.png", bbox_inches="tight")
        plt.close()
