"""
python english_data.py \
	"C:/Users/timoy/OneDrive/Desktop/SyllableTokenizationGPT/tokenizer" \
	"C:/Users/timoy/OneDrive/Desktop/SyllableTokenizationGPT/configs/tokenizer_configs/trained_eng_bpe_config.json" \
	--tasks syllables words g2p
"""

import requests
import json
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import random


# =========================
# DATA UNPACKING FUNCTIONS
# =========================

def unpack_and_syllabize(stored_path: str, bin_path: str, tokenizer, cross_val_counter: int):
	COMPLETE_SET_SIZE = 1000
	holdout_set_size = COMPLETE_SET_SIZE // cross_val_counter

	with open(stored_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	with_syllables = []

	for s in data:
		sentence = s["Sentence"]
		ids = tokenizer.tokenize_with_eos(sentence)
		syllables = s["syllable_count"]

		if syllables <= 30 and syllables > 1:
			ids.insert(-1, tokenizer.vocab.get("<1>"))
			ids.insert(-1, tokenizer.vocab.get(f"<{syllables}>"))
			with_syllables.append(ids)

	with_syllables = with_syllables[:COMPLETE_SET_SIZE]

	for i in range(cross_val_counter):
		test_slice = with_syllables[i * holdout_set_size:(i + 1) * holdout_set_size]
		train_slice = (
				with_syllables[:i * holdout_set_size] +
				with_syllables[(i + 1) * holdout_set_size:]
		)

		val_ratio = 0.1
		split_idx = int(len(train_slice) * (1 - val_ratio))

		train_set = train_slice[:split_idx]
		val_set = train_slice[split_idx:]

		put_into_file(test_slice, bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "test", i)
		put_into_file(train_set, bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "train", i)
		put_into_file(val_set, bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "val", i)


def unpack_and_wordize(stored_path: str, bin_path: str, tokenizer, cross_val_counter: int):
	COMPLETE_SET_SIZE = 1000
	holdout_set_size = COMPLETE_SET_SIZE // cross_val_counter

	with open(stored_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	with_words = []

	for s in data:
		sentence = s["Sentence"]
		ids = tokenizer.tokenize_with_eos(sentence)
		wc = len(sentence.split())

		if wc <= 30 and wc > 1:
			ids.insert(-1, tokenizer.vocab.get("<1>"))
			ids.insert(-1, tokenizer.vocab.get(f"<{wc}>"))
			with_words.append(ids)

	with_words = with_words[:COMPLETE_SET_SIZE]

	for i in range(cross_val_counter):
		test_slice = with_words[i * holdout_set_size:(i + 1) * holdout_set_size]
		train_slice = (
				with_words[:i * holdout_set_size] +
				with_words[(i + 1) * holdout_set_size:]
		)

		val_ratio = 0.1
		split_idx = int(len(train_slice) * (1 - val_ratio))

		train_set = train_slice[:split_idx]
		val_set = train_slice[split_idx:]

		put_into_file(test_slice, bin_path, tokenizer.language, tokenizer.paradigm, "word", "test", i)
		put_into_file(train_set, bin_path, tokenizer.language, tokenizer.paradigm, "word", "train", i)
		put_into_file(val_set, bin_path, tokenizer.language, tokenizer.paradigm, "word", "val", i)


def unpack_and_g2p(stored_path: str, bin_path: str, tokenizer, cross_val_counter: int):
	COMPLETE_SET_SIZE = 2000
	holdout_set_size = COMPLETE_SET_SIZE // cross_val_counter

	with open(stored_path, "r", encoding="utf-8") as f:
		data = [{"word": x, "ipa": y} for line in f for x, y in [line.strip().split('\t')]]

	random.seed(0)
	random.shuffle(data)

	g2peed = []

	for line in data[:COMPLETE_SET_SIZE]:
		ids = []
		ids.extend(tokenizer.tokenize(line["word"]))
		ids.append(tokenizer.vocab.get("<1>"))
		ids.extend(tokenizer.tokenize(line["ipa"]))
		ids.append(tokenizer.vocab.get("<EOS>"))
		g2peed.append(ids)

	for i in range(cross_val_counter):
		test_slice = g2peed[i * holdout_set_size:(i + 1) * holdout_set_size]
		train_slice = (
				g2peed[:i * holdout_set_size] +
				g2peed[(i + 1) * holdout_set_size:]
		)

		val_ratio = 0.1
		split_idx = int(len(train_slice) * (1 - val_ratio))

		train_set = train_slice[:split_idx]
		val_set = train_slice[split_idx:]

		put_into_file(test_slice, bin_path, tokenizer.language, tokenizer.paradigm, "g2p", "test", i)
		put_into_file(train_set, bin_path, tokenizer.language, tokenizer.paradigm, "g2p", "train", i)
		put_into_file(val_set, bin_path, tokenizer.language, tokenizer.paradigm, "g2p", "val", i)


# =========================
# FILE WRITING
# =========================

def write_datafile(filename, toks):
	assert len(toks) < 2 ** 31

	header = np.zeros(256, dtype=np.int32)
	header[0] = 20240520
	header[1] = 1
	header[2] = len(toks)

	if not isinstance(toks, np.ndarray) or toks.dtype != np.uint16:
		assert all(0 <= t < 2 ** 16 for t in toks)
		toks = np.array(toks, dtype=np.uint16)

	print(f"writing {len(toks):,} tokens to {filename}")
	with open(filename, "wb") as f:
		f.write(header.tobytes())
		f.write(toks.tobytes())


def put_into_file(ids, path: str, language: str, paradigm: str, task: str, split: str, cvc: int):
	shard_size = 10 ** 8
	shard_index = 0
	all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
	token_count = 0

	def _flatten(tokens):
		out = []
		for t in tokens:
			if isinstance(t, (list, tuple, np.ndarray)):
				out.extend(int(x) for x in t)
			else:
				out.append(int(t))
		return out

	ids = _flatten(ids)
	idx = 0
	remaining = len(ids)
	progress = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

	while remaining > 0:
		space = shard_size - token_count
		if space == 0:
			outname = os.path.join(
				path,
				f"{task}_{language}_{paradigm}_{split}_{cvc}_{shard_index:06d}.bin"
			)
			write_datafile(outname, all_tokens_np)
			shard_index += 1
			progress.close()
			progress = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
			token_count = 0
			space = shard_size

		take = min(space, remaining)
		all_tokens_np[token_count:token_count + take] = ids[idx:idx + take]

		token_count += take
		idx += take
		remaining -= take
		progress.update(take)

	if token_count > 0:
		outname = os.path.join(
			path,
			f"{task}_{language}_{paradigm}_{split}_{cvc}_{shard_index:06d}.bin"
		)
		write_datafile(outname, all_tokens_np[:token_count])


# =========================
# MAIN
# =========================

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("tokenizer_config", type=str)
	parser.add_argument("--cross_val_sets", type=int, default=10)
	parser.add_argument(
		"--tasks",
		nargs="+",
		default=["syllables", "words"],
		choices=["syllables", "words", "g2p"]
	)

	args = parser.parse_args()
	assert "eng" in args.tokenizer_config

	from Definitions.Tokenizer import TokenizerConfig, Tokenizer

	config = TokenizerConfig.load(args.tokenizer_config)
	tokenizer = Tokenizer(config)

	DATA_PATH = "Data/english_data"
	BIN_PATH = os.path.join(DATA_PATH, "bins")
	os.makedirs(BIN_PATH, exist_ok=True)

	# =========================
	# DATA SOURCES
	# =========================

	SYLLABLE_URL = (
		"https://raw.githubusercontent.com/asuvarna31/llm_phonology/main/syllable_counting/eval_sentence_counting.json"
	)

	G2P_URL = (
		"https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/eng-us.tsv"
	)
	phon_path = os.path.join(DATA_PATH, "phon_bench.json")
	g2p_path = os.path.join(DATA_PATH, "eng_g2p.tsv")

	if "syllables" in args.tasks or "words" in args.tasks:
		if not os.path.exists(phon_path):
			response = requests.get(SYLLABLE_URL)
			response.raise_for_status()

			data = [
				json.loads(line)
				for line in response.text.splitlines()
				if line.strip()
			]

			with open(phon_path, "w", encoding="utf-8") as f:
				json.dump(data, f, ensure_ascii=False, indent=2)
		else:
		    print(f"Using cached syllable data: {phon_path}")

		# =========================
		# FETCH + STORE G2P DATA (CACHE-AWARE)
		# =========================

	if "g2p" in args.tasks:
		if not os.path.exists(g2p_path):
			response = requests.get(G2P_URL)
			response.raise_for_status()

			with open(g2p_path, "w", encoding="utf-8") as f:
				f.write(response.text)
		else:
			print(f"Using cached G2P data: {g2p_path}")

	# =========================
	# TASK DISPATCH
	# =========================

	if "syllables" in args.tasks:
		unpack_and_syllabize(phon_path, BIN_PATH, tokenizer, args.cross_val_sets)

	if "words" in args.tasks:
		unpack_and_wordize(phon_path, BIN_PATH, tokenizer, args.cross_val_sets)

	if "g2p" in args.tasks:
		unpack_and_g2p(g2p_path, BIN_PATH, tokenizer, args.cross_val_sets)
