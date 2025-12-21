"""
python english_data.py "C:/Users/timoy/OneDrive/Desktop/SyllableTokenizationGPT/tokenizer" "C:/Users/timoy/OneDrive/Desktop/SyllableTokenizationGPT/configs/tokenizer_configs/trained_eng_bpe_config.json"
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json
import os
from urllib.parse import urljoin
import sys
import argparse
import numpy as np
from tqdm import tqdm

def unpack_and_syllabize(stored_path : str, bin_path : str, tokenizer, cross_val_counter : str):
    COMPLETE_SET_SIZE = 1000
    holdout_set_size = COMPLETE_SET_SIZE//cross_val_counter
    with open(stored_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with_syllables = []
    for s in data:
        sentence = s["Sentence"]
        ids = tokenizer.tokenize_with_eos(sentence)
        syllables = s["syllable_count"]
        if syllables <= 30 and syllables > 1:
            ids.insert(-1, tokenizer.vocab.get(f"<1>"))
            ids.insert(-1, tokenizer.vocab.get(f"<{syllables}>"))
            with_syllables.append(ids)
    i2c = {i:c for c,i in tokenizer.vocab.items()}
    for x in with_syllables[:20]:
        print(x)

        def join_tokens(tokens, encoding="utf-8", errors="strict"):
            out = bytearray()
            for t in tokens:
                if isinstance(t, str):
                    out.extend(t.encode(encoding, errors))
                else:
                    out.extend(t)
            return bytes(out)
        print(join_tokens([i2c[c] for c in x]))
    with_syllables = with_syllables[:COMPLETE_SET_SIZE]
    for i in range(cross_val_counter):
        test_slice = with_syllables[i*holdout_set_size:(i+1)*holdout_set_size]
        train_slice = with_syllables[:i * holdout_set_size] + with_syllables[(i + 1) * holdout_set_size:]
        val_ratio = 0.1
        split_idx = int(len(train_slice) * (1 - val_ratio))
        train_set = train_slice[:split_idx]
        val_set = train_slice[split_idx:]
        put_into_file(test_slice, bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "test", i)
        put_into_file(train_set, bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "train", i)
        put_into_file(val_set, bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "val", i)


def unpack_and_wordize(stored_path : str, bin_path : str, tokenizer, cross_val_counter : str):
    COMPLETE_SET_SIZE = 1000
    holdout_set_size = COMPLETE_SET_SIZE//cross_val_counter
    with open(stored_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with_words = []
    for s in data:
        sentence = s["Sentence"]
        ids = tokenizer.tokenize_with_eos(sentence)
        wc = len(sentence.split())
        if wc <= 30 and wc > 1:
            ids.insert(-1, tokenizer.vocab.get(f"<1>"))
            ids.insert(-1, tokenizer.vocab.get(f"<{wc}>"))
            with_words.append(ids)
    with_words = with_words[:COMPLETE_SET_SIZE]
    for i in range(cross_val_counter):
        test_slice = with_words[i*holdout_set_size:(i+1)*holdout_set_size]
        train_slice = with_words[:i * holdout_set_size] + with_words[(i + 1) * holdout_set_size:]
        val_ratio = 0.1
        split_idx = int(len(train_slice) * (1 - val_ratio))
        train_set = train_slice[:split_idx]
        val_set = train_slice[split_idx:]
        put_into_file(test_slice, bin_path, tokenizer.language, tokenizer.paradigm, "word", "test", i)
        put_into_file(train_set, bin_path, tokenizer.language, tokenizer.paradigm, "word", "train", i)
        put_into_file(val_set, bin_path, tokenizer.language, tokenizer.paradigm, "word", "val", i)

def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2 ** 31, "token count too large"  # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(toks)  # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2 ** 16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def put_into_file(ids, path : str, language : str, paradigm : str, task : str, split : str, cvc : int):
    """
    Adapted from the NanoGPT Speedrun code
    Script kiddie, GO!
    """
    shard_size = 10 ** 8
    shard_index = 0 # realistically, it will only be zero
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0

    def _flatten_to_ints(tokens):
        if tokens and isinstance(tokens[0], (list, tuple, np.ndarray)):
            flat = []
            for t in tokens:
                if isinstance(t, (list, tuple, np.ndarray)):
                    flat.extend(int(x) for x in t)
                else:
                    flat.append(int(t))
            return flat
        return [int(x) for x in tokens]

    ids = _flatten_to_ints(ids)
    idx = 0
    remaining = len(ids)

    def _new_progress(shard_idx):
        return tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}")

    progress_bar = _new_progress(shard_index)

    while remaining > 0:
        space = shard_size - token_count
        if space == 0:
            # shard full: write and open new one
            outname = os.path.join(
                path,
                f"{task}_{language}_{paradigm}_{split}_{cvc}_{shard_index:06d}.bin"
            )
            write_datafile(outname, all_tokens_np)
            shard_index += 1
            progress_bar.close()
            progress_bar = _new_progress(shard_index)
            token_count = 0
            space = shard_size
        take = min(space, remaining)
        all_tokens_np[token_count:token_count + take] = ids[idx: idx + take]
        token_count += take
        idx += take
        remaining -= take
        progress_bar.update(take)

    if token_count != 0:
        outname = os.path.join(
            path,
            f"{task}_{language}_{paradigm}_{split}_{cvc}_{shard_index:06d}.bin"
        )
        # write only the portion filled
        write_datafile(outname, all_tokens_np[:token_count])



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("tokenizers_folder", type=str, help="Absolute Path to tokenizer folder")
    arg_parser.add_argument("tokenizer_config", type=str, help="trained config path")
    arg_parser.add_argument("--cross_val_sets", type=int, default=10)

    args = arg_parser.parse_args()

    assert "eng" in args.tokenizer_config, "ENGLISH CONFIG PLS"

    tokenizer_directory = os.path.abspath(args.tokenizers_folder)
    sys.path.insert(0, tokenizer_directory)
    from Tokenizer import TokenizerConfig, Tokenizer
    TOKENIZER_CONFIG = TokenizerConfig.load(args.tokenizer_config)
    tokenizer = Tokenizer(TOKENIZER_CONFIG)
    DATA_PATH = "english_data"
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
        BIN_PATH = os.path.join(DATA_PATH, "bins")
        if not os.path.exists(BIN_PATH):
            os.makedirs(BIN_PATH)

    url = "https://raw.githubusercontent.com/asuvarna31/llm_phonology/main/syllable_counting/eval_sentence_counting.json"
    response = requests.get(url)
    response.raise_for_status()  # raise an error if the request failed
    data = [
        json.loads(line)
        for line in response.text.splitlines()
        if line.strip()
    ]
    BIN_PATH = os.path.join(DATA_PATH, "bins")
    stored_path = f"{DATA_PATH}/phon_bench.json"
    with open(stored_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    unpack_and_syllabize(stored_path, BIN_PATH, tokenizer, args.cross_val_sets)
    unpack_and_wordize(stored_path, BIN_PATH, tokenizer, args.cross_val_sets)
