import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

compiler = {}

# Read files and store data
for file in os.listdir("."):
	if file.endswith(".txt"):
		task, language, paradigm, cvc = file.rstrip(".txt").split("_")
		with open(file, "r") as f:
			prop, hits, total = [float(x) for x in f.readline().strip().split(',')[-3:]]

		# Initialize nested dictionaries if they don't exist
		compiler.setdefault(task, {})
		compiler[task].setdefault(language, {})
		compiler[task][language].setdefault(paradigm, [])

		# Append the CVC data as a dictionary
		compiler[task][language][paradigm].append({
			"cvc": cvc,
			"hits": hits,
			"totals": total
		})

os.makedirs("graphs", exist_ok=True)
tasks = compiler.keys()

for task in tasks:
	languages = compiler[task].keys()

	for language in languages:
		paradigms = compiler[task][language].keys()
		avg_props = []
		conf_intervals = []
		labels = []
		ns = []  # total of totals per paradigm

		for paradigm in paradigms:
			cvc_list = [cvc_data for cvc_data in compiler[task][language][paradigm] if cvc_data["totals"] != 0]
			total_total = sum(cvc_data["totals"] for cvc_data in cvc_list)
			ns.append(total_total)

			props_list = [cvc_data["hits"] / cvc_data["totals"] for cvc_data in cvc_list]

			# Compute mean proportion
			mean_prop = np.mean(props_list) if props_list else 0

			# Compute 95% confidence interval
			if len(props_list) > 1:
				conf = stats.t.interval(
					0.95,
					len(props_list) - 1,
					loc=np.mean(props_list),
					scale=stats.sem(props_list)
				)
				error = (conf[1] - conf[0]) / 2
			else:
				error = 0

			avg_props.append(mean_prop)
			conf_intervals.append(error)
			labels.append(paradigm)

		# Plot proportion per paradigm
		plt.figure(figsize=(8, 5))
		bars = plt.bar(labels, avg_props, yerr=conf_intervals, capsize=5)
		plt.ylabel("Proportion of Hits")
		plt.title(f"Task: {task} | Language: {language}")
		plt.ylim(0, 0.2)

		# Annotate total totals (n) on top of each bar
		for bar, n in zip(bars, ns):
			height = bar.get_height()
			plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"n={int(n)}", ha='center', va='bottom')

		plt.savefig(f"graphs/{task}_{language}_proportions.png", bbox_inches="tight")
		plt.close()