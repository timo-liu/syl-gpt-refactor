#!/usr/bin/env python
"""
english_data.py

Prepares English data for syllable, word, and g2p tasks using GPT-2 tokenizer.
Last 30 GPT-2 tokens are mapped as <1> - <30>.
"""

import requests
import json
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import random
from transformers import AutoTokenizer

# =========================
# DATA UNPACKING FUNCTIONS
# =========================

def unpack_and_syllabize(stored_path, bin_path, tokenizer, token_map, cross_val_counter):
    COMPLETE_SET_SIZE = 1000
    holdout_set_size = COMPLETE_SET_SIZE // cross_val_counter

    with open(stored_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with_syllables = []

    for s in data:
        sentence = s["Sentence"]
        ids = tokenizer.encode(sentence, add_special_tokens=True)
        syllables = s["syllable_count"]

        if 1 < syllables <= 30:
            # Map to GPT-2 special tokens
            ids.insert(-1, token_map[1])           # start
            ids.insert(-1, token_map[syllables])  # syllable count
            with_syllables.append(ids)

    with_syllables = with_syllables[:COMPLETE_SET_SIZE]

    for i in range(cross_val_counter):
        test_slice = with_syllables[i * holdout_set_size:(i + 1) * holdout_set_size]
        train_slice = with_syllables[:i * holdout_set_size] + with_syllables[(i + 1) * holdout_set_size:]

        val_ratio = 0.1
        split_idx = int(len(train_slice) * (1 - val_ratio))

        train_set = train_slice[:split_idx]
        val_set = train_slice[split_idx:]

        put_into_file(test_slice, bin_path, "eng", "gpt2", "syllables", "test", i)
        put_into_file(train_set, bin_path, "eng", "gpt2", "syllables", "train", i)
        put_into_file(val_set, bin_path, "eng", "gpt2", "syllables", "val", i)


def unpack_and_wordize(stored_path, bin_path, tokenizer, token_map, cross_val_counter):
    COMPLETE_SET_SIZE = 1000
    holdout_set_size = COMPLETE_SET_SIZE // cross_val_counter

    with open(stored_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with_words = []

    for s in data:
        sentence = s["Sentence"]
        ids = tokenizer.encode(sentence, add_special_tokens=True)
        wc = len(sentence.split())

        if 1 < wc <= 30:
            ids.insert(-1, token_map[1])
            ids.insert(-1, token_map[wc])
            with_words.append(ids)

    with_words = with_words[:COMPLETE_SET_SIZE]

    for i in range(cross_val_counter):
        test_slice = with_words[i * holdout_set_size:(i + 1) * holdout_set_size]
        train_slice = with_words[:i * holdout_set_size] + with_words[(i + 1) * holdout_set_size:]

        val_ratio = 0.1
        split_idx = int(len(train_slice) * (1 - val_ratio))

        train_set = train_slice[:split_idx]
        val_set = train_slice[split_idx:]

        put_into_file(test_slice, bin_path, "eng", "gpt2", "word", "test", i)
        put_into_file(train_set, bin_path, "eng", "gpt2", "word", "train", i)
        put_into_file(val_set, bin_path, "eng", "gpt2", "word", "val", i)


def unpack_and_g2p(stored_path, bin_path, tokenizer, cross_val_counter):
    COMPLETE_SET_SIZE = 2000
    holdout_set_size = COMPLETE_SET_SIZE // cross_val_counter

    with open(stored_path, "r", encoding="utf-8") as f:
        data = [{"word": x, "ipa": y} for line in f for x, y in [line.strip().split("\t")]]

    random.seed(0)
    random.shuffle(data)

    g2peed = []

    for line in data[:COMPLETE_SET_SIZE]:
        ids = tokenizer.encode(line["word"], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)  # EOS
        ids.extend(tokenizer.encode(line["ipa"], add_special_tokens=False))
        ids.append(tokenizer.eos_token_id)
        g2peed.append(ids)

    for i in range(cross_val_counter):
        test_slice = g2peed[i * holdout_set_size:(i + 1) * holdout_set_size]
        train_slice = g2peed[:i * holdout_set_size] + g2peed[(i + 1) * holdout_set_size:]

        val_ratio = 0.1
        split_idx = int(len(train_slice) * (1 - val_ratio))

        train_set = train_slice[:split_idx]
        val_set = train_slice[split_idx:]

        put_into_file(test_slice, bin_path, "eng", "gpt2", "g2p", "test", i)
        put_into_file(train_set, bin_path, "eng", "gpt2", "g2p", "train", i)
        put_into_file(val_set, bin_path, "eng", "gpt2", "g2p", "val", i)


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


def put_into_file(ids, path, language, paradigm, task, split, cvc):
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
            outname = os.path.join(path, f"{task}_{language}_{paradigm}_{split}_{cvc}_{shard_index:06d}.bin")
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
        outname = os.path.join(path, f"{task}_{language}_{paradigm}_{split}_{cvc}_{shard_index:06d}.bin")
        write_datafile(outname, all_tokens_np[:token_count])


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_pipeline", action="store_true",
                        help="Override all args and run full English GPT-2 pipeline")
    parser.add_argument("--cross_val_sets", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=["syllables", "words", "g2p"],
                        choices=["syllables", "words", "g2p"])
    args = parser.parse_args()

    DATA_PATH = "Data/english_data"
    BIN_PATH = os.path.join(DATA_PATH, "bins")
    os.makedirs(BIN_PATH, exist_ok=True)

    # URLs
    SYLLABLE_URL = "https://raw.githubusercontent.com/asuvarna31/llm_phonology/main/syllable_counting/eval_sentence_counting.json"
    G2P_URL = "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/eng-us.tsv"
    phon_path = os.path.join(DATA_PATH, "phon_bench.json")
    g2p_path = os.path.join(DATA_PATH, "eng_g2p.tsv")

    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Map last 30 GPT-2 tokens to <1>-<30>
    special_tokens = [f"<{i}>" for i in range(1, 31)]
    special_token_ids = list(range(tokenizer.vocab_size - 30, tokenizer.vocab_size))
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    token_map = {i: tid for i, tid in zip(range(1, 31), special_token_ids)}

    # =========================
    # FETCH DATA (CACHE AWARE)
    # =========================
    if not os.path.exists(phon_path):
        response = requests.get(SYLLABLE_URL)
        response.raise_for_status()
        data = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        with open(phon_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        print(f"Using cached syllable data: {phon_path}")

    if not os.path.exists(g2p_path):
        response = requests.get(G2P_URL)
        response.raise_for_status()
        with open(g2p_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    else:
        print(f"Using cached G2P data: {g2p_path}")

    # =========================
    # RUN PIPELINE
    # =========================
    if args.run_pipeline:
        print("Running full English GPT-2 pipeline (syllables, words, g2p)")
        unpack_and_syllabize(phon_path, BIN_PATH, tokenizer, token_map, args.cross_val_sets)
        unpack_and_wordize(phon_path, BIN_PATH, tokenizer, token_map, args.cross_val_sets)
        unpack_and_g2p(g2p_path, BIN_PATH, tokenizer, args.cross_val_sets)
    else:
        for task in args.tasks:
            if task == "syllables":
                unpack_and_syllabize(phon_path, BIN_PATH, tokenizer, token_map, args.cross_val_sets)
            elif task == "words":
                unpack_and_wordize(phon_path, BIN_PATH, tokenizer, token_map, args.cross_val_sets)
            elif task == "g2p":
                unpack_and_g2p(g2p_path, BIN_PATH, tokenizer, args.cross_val_sets)