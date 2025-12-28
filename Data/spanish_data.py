"""
python spanish_data.py \
    "C:/path/to/tokenizer" \
    "C:/path/to/configs/trained_span_bpe_config.json" \
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
import time
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from syltippy import syllabize

# =========================
# STORY SCRAPING (SPANISH)
# =========================

BASE_URL = "https://www.fluentwithstories.com"
START_URL = f"{BASE_URL}/stories/es"


def extract_story_links(start_url=START_URL):
    all_links = []
    next_url = start_url
    seen_pages = set()

    while next_url and next_url not in seen_pages:
        print(f"Fetching story list: {next_url}")
        seen_pages.add(next_url)

        resp = requests.get(next_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.select("div.w-dyn-item a.story_card_component.w-inline-block"):
            href = a.get("href")
            if href and href.startswith("/stories/es/"):
                all_links.append(urljoin(BASE_URL, href))

        next_link = soup.select_one("a.w-pagination-next")
        if next_link and next_link.get("href"):
            next_url = urljoin(start_url, next_link["href"])
        else:
            next_url = None

    return sorted(set(all_links))


def parse_story(url):
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    m = re.search(r"/stories/es/([ab]\d)-", url)
    level = m.group(1).upper() if m else "UNKNOWN"

    content_div = soup.select_one('div.story-rich-text.w-richtext[lang="es"]')

    sentences = []
    if content_div:
        paragraphs = [p.get_text(" ", strip=True) for p in content_div.find_all("p")]
        text = " ".join(paragraphs)
        for sent in re.split(r'(?<=[\.\?!¡¿])\s+', text):
            sent = sent.strip()
            if sent:
                sentences.append(sent)

    return level, sentences


def scrape_and_store_spanish(DATA_PATH):
    out_path = os.path.join(DATA_PATH, "fluentwithstories_spanish.json")

    if os.path.exists(out_path):
        print(f"Using cached Spanish stories: {out_path}")
        return out_path

    all_sentences = []
    links = extract_story_links()
    print(f"Found {len(links)} total stories.")

    for i, link in enumerate(links, start=1):
        print(f"[{i}/{len(links)}] Scraping: {link}")
        level, sents = parse_story(link)
        for s in sents:
            all_sentences.append({"level": level, "sentence": s})
        time.sleep(0.5)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_sentences)} sentences to {out_path}")
    return out_path


# =========================
# DATA UNPACKING
# =========================

def unpack_and_syllabize(stored_path, bin_path, tokenizer, cross_val_counter):
    COMPLETE_SET_SIZE = 1000
    holdout = COMPLETE_SET_SIZE // cross_val_counter

    with open(stored_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for s in data:
        sent = s["sentence"]
        ids = tokenizer.tokenize_with_eos(sent)
        syls = len(syllabize(sent)[0])

        if 1 < syls <= 30:
            ids.insert(-1, tokenizer.vocab.get("<1>"))
            ids.insert(-1, tokenizer.vocab.get(f"<{syls}>"))
            out.append(ids)

    out = out[:COMPLETE_SET_SIZE]

    for i in range(cross_val_counter):
        test = out[i * holdout:(i + 1) * holdout]
        train = out[:i * holdout] + out[(i + 1) * holdout:]

        split = int(len(train) * 0.9)
        put_into_file(test, bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "test", i)
        put_into_file(train[:split], bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "train", i)
        put_into_file(train[split:], bin_path, tokenizer.language, tokenizer.paradigm, "syllables", "val", i)


def unpack_and_wordize(stored_path, bin_path, tokenizer, cross_val_counter):
    COMPLETE_SET_SIZE = 1000
    holdout = COMPLETE_SET_SIZE // cross_val_counter

    with open(stored_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for s in data:
        sent = s["sentence"]
        wc = len(sent.split())
        ids = tokenizer.tokenize_with_eos(sent)

        if 1 < wc <= 30:
            ids.insert(-1, tokenizer.vocab.get("<1>"))
            ids.insert(-1, tokenizer.vocab.get(f"<{wc}>"))
            out.append(ids)

    out = out[:COMPLETE_SET_SIZE]

    for i in range(cross_val_counter):
        test = out[i * holdout:(i + 1) * holdout]
        train = out[:i * holdout] + out[(i + 1) * holdout:]

        split = int(len(train) * 0.9)
        put_into_file(test, bin_path, tokenizer.language, tokenizer.paradigm, "word", "test", i)
        put_into_file(train[:split], bin_path, tokenizer.language, tokenizer.paradigm, "word", "train", i)
        put_into_file(train[split:], bin_path, tokenizer.language, tokenizer.paradigm, "word", "val", i)


def unpack_and_g2p(stored_path, bin_path, tokenizer, cross_val_counter):
    COMPLETE_SET_SIZE = 2000
    holdout = COMPLETE_SET_SIZE // cross_val_counter

    with open(stored_path, "r", encoding="utf-8") as f:
        data = [{"word": x, "ipa": y} for line in f for x, y in [line.strip().split("\t")]]

    random.seed(0)
    random.shuffle(data)

    g2p = []
    for d in data[:COMPLETE_SET_SIZE]:
        ids = []
        ids.extend(tokenizer.tokenize(d["word"]))
        ids.append(tokenizer.vocab.get("<1>"))
        ids.extend(tokenizer.tokenize(d["ipa"]))
        ids.append(tokenizer.vocab.get("<EOS>"))
        g2p.append(ids)

    for i in range(cross_val_counter):
        test = g2p[i * holdout:(i + 1) * holdout]
        train = g2p[:i * holdout] + g2p[(i + 1) * holdout:]

        split = int(len(train) * 0.9)
        put_into_file(test, bin_path, tokenizer.language, tokenizer.paradigm, "g2p", "test", i)
        put_into_file(train[:split], bin_path, tokenizer.language, tokenizer.paradigm, "g2p", "train", i)
        put_into_file(train[split:], bin_path, tokenizer.language, tokenizer.paradigm, "g2p", "val", i)


# =========================
# BIN WRITING
# =========================

def write_datafile(filename, toks):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(toks)

    toks = np.array(toks, dtype=np.uint16)
    print(f"writing {len(toks):,} tokens to {filename}")

    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def put_into_file(ids, path, language, paradigm, task, split, cvc):
    shard_size = 10 ** 8
    all_tokens = []
    for x in ids:
        if isinstance(x, (list, tuple, np.ndarray)):
            all_tokens.extend(int(t) for t in x)
        else:
            all_tokens.append(int(x))

    outname = os.path.join(
        path, f"{task}_{language}_{paradigm}_{split}_{cvc}_000000.bin"
    )
    write_datafile(outname, all_tokens)


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

    assert "span" in args.tokenizer_config

    from Definitions.Tokenizer import TokenizerConfig, Tokenizer

    tokenizer = Tokenizer(TokenizerConfig.load(args.tokenizer_config))

    DATA_PATH = "Data/spanish_data"
    BIN_PATH = os.path.join(DATA_PATH, "bins")
    os.makedirs(BIN_PATH, exist_ok=True)

    G2P_URL = "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/spa.tsv"
    g2p_path = os.path.join(DATA_PATH, "spa_g2p.tsv")

    story_path = scrape_and_store_spanish(DATA_PATH)

    if "g2p" in args.tasks:
        if not os.path.exists(g2p_path):
            r = requests.get(G2P_URL)
            r.raise_for_status()
            with open(g2p_path, "w", encoding="utf-8") as f:
                f.write(r.text)
        else:
            print(f"Using cached G2P data: {g2p_path}")

    if "syllables" in args.tasks:
        unpack_and_syllabize(story_path, BIN_PATH, tokenizer, args.cross_val_sets)

    if "words" in args.tasks:
        unpack_and_wordize(story_path, BIN_PATH, tokenizer, args.cross_val_sets)

    if "g2p" in args.tasks:
        unpack_and_g2p(g2p_path, BIN_PATH, tokenizer, args.cross_val_sets)
