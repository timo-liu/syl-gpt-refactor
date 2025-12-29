# region imports
import os
import json
import re
import ast
import math
from dataclasses import dataclass, field, MISSING
from typing import Any, Dict, List, Union, Tuple
from string import hexdigits
import numpy as np
from tqdm import tqdm
from syltippy import syllabize
# endregion imports

# region global
def _utf8_bytes_ids(self, vocab, s):
    bs = s.encode("utf-8")
    out = []
    for b in bs:
        bid = vocab.get(bytes([b]))
        if bid is None:
            raise KeyError(f"Byte {b:#04x} not in vocab")
        out.append(bid)
    return out
def utf8_bytes_ids(self, s: str) -> List[int]:
            bs = s.encode("utf-8", errors="strict")
            out = []
            for b in bs:
                bid = self.vocab.get(bytes([b]))
                if bid is None:
                    # if you ever allow a fallback id, use it here; otherwise raise
                    raise KeyError(f"Byte {b:#04x} not in vocab")
                out.append(bid)
            return out
def _tok_worker(line):
    paradigm = _TOK_WORKER["paradigm"]
    vocab = _TOK_WORKER["vocab"]
    eos_id = _TOK_WORKER["eos_id"]

    if paradigm in ("bpe", "blbpe", "uni"):
        ids = _TOK_WORKER["sp"].encode(line, out_type=int)
    elif paradigm == "morf":
        ids = _encode_morf(vocab, line)
    else:
        ids = _utf8_bytes_ids(self, vocab, line)

    ids.append(eos_id)
    return ids
# endregion global

# region TokenizerConfig
@dataclass
class TokenizerConfig:
    language: str
    paradigm: str
    vocab: Dict[Any, int]           # keys can be bytes or special strings
    vocab_size: int
    initial_vocab: Dict[Any, int] = field(default_factory=dict)
    merge_to: int = 0

    @classmethod
    def load(cls, config_path: str = None, **kwargs) -> "TokenizerConfig":
        cfg = {}

        if config_path:
            assert os.path.exists(config_path), "Wrong path dumbass"
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        cfg.update(kwargs)

        required = {
            name for name, f in cls.__dataclass_fields__.items()
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required - cfg.keys()
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        def looks_like_hex(s: str) -> bool:
            compact = ''.join(ch for ch in s if not ch.isspace())
            return bool(compact) and len(compact) % 2 == 0 and all(c in hexdigits for c in compact)

        def key_from_json(k):
            # Handle Python-style bytes literals like: "b'\\xff'"
            if isinstance(k, str) and (k.startswith("b'") or k.startswith('b"')):
                try:
                    val = ast.literal_eval(k)
                    if isinstance(val, (bytes, bytearray)):
                        return bytes(val)
                except (SyntaxError, ValueError):
                    pass

            # If the JSON key looks like it contains escape sequences such as \xFF, \u1234, \U00012345, or common escapes
            if isinstance(k, str) and re.search(r'\\x[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}|\\[nrt\\\'"]', k):
                try:
                    # Turn the literal backslash escapes into actual characters:
                    # e.g. "\\xff" -> "\xff" (a one-character string with codepoint 255)
                    unescaped = bytes(k, 'utf-8').decode('unicode_escape')

                    # Prefer returning a bytes object:
                    # - If all codepoints are < 256, latin-1 maps them 1:1 to bytes (ideal for original bytes intent).
                    # - Otherwise fall back to utf-8 encoding (multi-byte).
                    try:
                        return unescaped.encode('latin-1')
                    except UnicodeEncodeError:
                        return unescaped.encode('utf-8')
                except Exception:
                    # If something goes wrong, fall through to other heuristics
                    pass

            # Keep special-angle tokens like "<...>" as-is
            if isinstance(k, str) and k.startswith('<') and k.endswith('>'):
                return k

            # Single-character strings: encode to utf-8 (previous behavior)
            if isinstance(k, str) and len(k) == 1:
                return k.encode('utf-8')

            # If the string is purely hex (no whitespace), accept it as hex bytes
            if isinstance(k, str):
                compact = ''.join(ch for ch in k if not ch.isspace())

                # NEVER hex-decode short alphabetic tokens
                if len(compact) < 4:
                    return k

                if (
                        len(compact) % 2 == 0
                        and all(c in hexdigits for c in compact)
                ):
                    return bytes.fromhex(compact)

            # Default: return as-is (likely a normal string key)
            return k

        def canon_dict(d):
            return {key_from_json(k): int(v) for k, v in d.items()}

        if "initial_vocab" in cfg:
            cfg["initial_vocab"] = canon_dict(cfg["initial_vocab"])
        if "vocab" in cfg and cfg["vocab"] is not None:
            print("loading dict")
            cfg["vocab"] = canon_dict(cfg["vocab"])

        if "vocab_size" in cfg:
            cfg["vocab_size"] = int(cfg["vocab_size"])
        if "merge_to" in cfg:
            cfg["merge_to"] = int(cfg["merge_to"])

        return cls(**cfg)
# endregion TokenizerConfig

# region Tokenizer
class Tokenizer():
    def __init__(self,
                 config : TokenizerConfig
                 ):
        # let's define some instance attributes
        self.autotokenizer = False    # is this tokenizer running off an autotokenizer?
        self.eats_files = False       # does this tokenizer run off a backbone that ingest files internally?
        self.language = ""            # not going to "assume" a language lmao
        self.paradigm = ""            # neither shall I assume a paradigm
        self.trainable = True         # just a sanity check to ensure that je suis utilise un model untrained
        self.vocab = {}               # el vocab
        self.vocab_size = 0           # post training, what size is the vocab?
        self.i2c = {}                 # ez i2c
        self.initial_vocab_size = 0   # this will be subtracted from the final merge size for non backboners that take this into account
        self.initial_vocab = None     # this is important for sentencepiece so auuuuuugh
        self.uses_syllabifier = False # does this model use the syllabifier model?
        self.morfessor_io = None      # Morfessorio


        # refactor into two paths: the autotokenizers, and the self-defined

        # autotokenizers
        if config.paradigm in ["gpt-2"]:
            from transformers import AutoTokenizer
            self.trainable = False
            self.language = config.language
            self.autotokenizer = AutoTokenizer.from_pretrained(self.paradigm)

        elif config.paradigm in ["bpe", "syl", "morf", "uni"]:
            # this config has clearly been trained already.
            if config.vocab is not None:
                self.trainable = False
                self.vocab = config.vocab
                self.vocab_size = len(self.vocab)
                self.vocab_size = config.vocab_size
                self.i2c = {i:c for c,i in self.vocab.items()}
            else:
                # this buffoon needs to be trained
                # not actually going to set vocab to intial vocab. Just going to set a final merge to
                self.initial_vocab_size = len(config.initial_vocab)

        # setting what should be universal attributes between configs
        self.initial_vocab = config.initial_vocab
        self.paradigm = config.paradigm
        self.language = config.language

        # splitting into section that require some backbone
        if self.paradigm == "syl":
            if self.language == "eng":
                from eng_syl_torch import Syllabifier as S
                self.uses_syllabifier = True
                self.syllabifier = S()
                pass

        elif self.paradigm == 'morf':
            import morfessor
            self.eats_files = True # morfessor does in fact eat files
            self.morfessor_io = morfessor.MorfessorIO()
            if os.path.exists(f"{self.paradigm}_{self.language}_morfessor.bin"):
                self.morf_model = self.morfessor_io.read_binary_model_file(f"{self.paradigm}_{self.language}_morfessor.bin")

        # our sentencepiece backboners
        elif self.paradigm in ["uni", "bpe"]:
            import sentencepiece as spm
            self.eats_files = True # this also eats files
            if os.path.exists(f'{self.language}_{self.paradigm}.model'):
                self.sp = spm.SentencePieceProcessor(model_file=f'{self.language}_{self.paradigm}.model')

    def decode(self, coded: List[int], debug=False):
        if self.paradigm == "syl":
            parts = []
            byte_buffer = bytearray()

            for i in coded:
                tok = self.i2c[i]

                if isinstance(tok, bytes):
                    byte_buffer.extend(tok)
                else:
                    if byte_buffer:
                        parts.append(byte_buffer.decode("utf-8", errors="replace"))
                        byte_buffer.clear()
                    parts.append(str(tok))

            if byte_buffer:
                parts.append(byte_buffer.decode("utf-8", errors="replace"))

            if not debug:
                return "".join(parts)
            else:
                return parts
        else:
            if self.sp:
                return self.sp.decode(coded)
            assert False, "YOU HAVE NOTHING"

    def train(self,
              text_corpus_path : str,
              merge_to : int = 0
              ):
        """
        :param text_corpus_path:  path to training corpus
        :param merge_to: how many types the resulting vocab should have if bpe/ blbpe
        :return:
        """
        assert self.trainable, "Need an empty vocab"
        assert merge_to > 256 if self.paradigm in ["bpe"] else True, "Need some vocab size to merge to greater than 256"

        # out merge to should be merge_to - initial_vocab size
        # preprocessing is specific to each function sooooooo
        # training
        num_tokens = 0

        if self.eats_files:
            # handling all eaters in here
            # these 孩子们用 le sentencepice backbone
            if self.paradigm in ["bpe", "uni"]:
                self.sentencepiece_backboner_training(text_corpus_path, final_vocab_size=merge_to)
            if self.paradigm == "morf":
                self.morfessor_training(text_corpus_path, merge_to)

        else:
            # ugh, handling preprocessing locally makes the eater distinction useless
            if self.paradigm == "syl":
                num_tokens = self.syllable_training(text_corpus_path, merge_to)
        self.vocab_size = len(self.vocab)
        self.trainable = False

        training_summary = {
            "num_tokens" : num_tokens,
            "vocab_size" : self.vocab_size
        }

        with open(f"{self.paradigm}_{self.language}_summary.json", 'w') as f:
            json.dump(training_summary, f)

    def syllable_training(self, path, merge_to):
        def yielder(path, batch_size):
            with open(path, 'r', encoding='utf-8') as f:
                batch = []
                j = 0
                for line in f:
                    line = re.findall(r'\s*\S+', re.sub(r'[\r\n\t]+', ' ', line).strip())
                    if j < batch_size:
                        batch.append(line)
                        j += 1
                    else:
                        return_batch = batch
                        batch = []
                        j = 0
                        yield return_batch

        from collections import Counter
        token_counter = Counter()
        if self.language == "eng":
            batch_size = 12600
            for batch in tqdm(yielder(path, batch_size)):
                outputs = self.syllabifier.machine_syllabify(batch, return_list=True)
                token_counter.update([x for l in outputs for x in l[0]])
        elif self.language == "span":
            batch_size = 12600
            for batch in tqdm(yielder(path, batch_size)):
                token_counter.update([x for line in batch for w in line for x in syllabize(w)[0]])
        self.vocab = {v:i for i,v in enumerate([x[0] for x in token_counter.most_common(merge_to - self.initial_vocab_size)], start = self.initial_vocab_size + 1)}
        self.vocab = self.vocab | self.initial_vocab
        return token_counter.total()

    def morfessor_training(self, corpus, merge_to):
        import morfessor
        """
        :param corpus: path to corpus
        :return:
        """
        train_data = list(self.morfessor_io.read_corpus_file(corpus))
        model_logtokens = morfessor.BaselineModel(corpusweight=5.0)

        def log_func(x):
            return int(round(math.log(x + 1, 2)))

        model_logtokens.load_data(train_data, count_modifier=log_func)
        model_logtokens.train_batch()
        self.morfessor_io.write_binary_file(f"{self.paradigm}_{self.language}_morfessor.bin", model_logtokens)
        self.morf_model = model_logtokens
        morph_counts = dict(model_logtokens.get_constructions())
        sorted_morphs = sorted(morph_counts.items(), key=lambda x: x[1], reverse=True)

        # how many new entries you can still add
        remaining = merge_to - self.initial_vocab_size
        self.vocab = self.intial_vocab
        if remaining > 0:
            for morph, _ in sorted_morphs:
                if morph not in self.vocab:
                    self.vocab[morph] = len(self.vocab) + 1
                    remaining -= 1
                    if remaining <= 0:
                        break

    def sentencepiece_backboner_training(self, path, final_vocab_size):
        import sentencepiece as spm
        # preprocess whatever orpus we have into little pieces
        chunk_lines = 500_000  # adjust per memory
        prefix = "tokenization"
        out_dir = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        chunk_paths = []

        # --- Split the huge file into smaller ones ---
        print(f"Splitting large corpus: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            chunk_idx = 0
            lines = []
            for i, line in enumerate(tqdm(f, desc="Reading corpus")):
                lines.append(line)
                if len(lines) >= chunk_lines:
                    chunk_file = os.path.join(out_dir, f"{prefix}_{chunk_idx}.txt")
                    with open(chunk_file, "w", encoding="utf-8") as cf:
                        cf.writelines(lines)
                    chunk_paths.append(chunk_file)
                    lines.clear()
                    chunk_idx += 1
            # write remainder
            if lines:
                chunk_file = os.path.join(out_dir, f"{prefix}_{chunk_idx}.txt")
                with open(chunk_file, "w", encoding="utf-8") as cf:
                    cf.writelines(lines)
                chunk_paths.append(chunk_file)
        spm.SentencePieceTrainer.train(input = ",".join(chunk_paths),
                                       model_prefix = f"{self.language}_{self.paradigm}",
                                       vocab_size = final_vocab_size,
                                       user_defined_symbols = list(self.initial_vocab.keys()),
                                       model_type = {"uni" : "unigram", "bpe" : "bpe"}.get(self.paradigm, None),
                                       byte_fallback = True,
                                       train_extremely_large_corpus = True
                                       )
        sp = spm.SentencePieceProcessor(model_file=f'{self.language}_{self.paradigm}.model')
        self.vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

    def save_tokenizer(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            info = {
                "language" : self.language,
                "paradigm" : self.paradigm,
                "vocab" : { str(b) : i for b,i in self.vocab.items()},
                "vocab_size" : self.vocab_size
            }
            json.dump(info, f, indent=4)

    def tokenize(self, text, debug : bool = False):
        assert not self.trainable, "This needs to be trained, or load a tokenizer with a vocab"
        if self.paradigm == "syl" and self.language == "eng":
            text = re.findall(r'\s*\S+', re.sub(r'[\r\n\t]+', ' ', text).strip())
            segmented = self.syllabifier.machine_syllabify(text, return_list=True)
            if not debug:
                return [x for w in segmented for s in w for x in self.encode(s)]
            else:
                return [
                    self.encode(s, debug=debug)
                    for w in segmented
                    for s in w
                    ]
        if self.paradigm == "syl" and self.language == "span":
            text = re.findall(r'\s*\S+', re.sub(r'[\r\n\t]+', ' ', text).strip())
            segmented = [x for w in text for x in syllabize(w)[0]]
            if not debug:
                return [
                    y
                    for w in segmented
                    for y in
                    self.encode(w, debug=debug)
                ]
            else:
                return [
                    self.encode(w, debug=debug)
                    for w in segmented
                ]
        if self.paradigm in ["uni", "bpe"]:
            return self.sp.encode(text, out_type=int)
        if self.paradigm == "morf":
            return self.encode(text)

    def encode(self, token: str, debug: bool = False):
        """
        Encode a token to ids. If a unit isn't in vocab, fall back to UTF-8 bytes
        (each byte mapped to its id). Never drop information.
        """
        assert not self.trainable, "This needs to be trained, or load a tokenizer with a vocab"

        if self.paradigm == "syl":
            # Expect syllables as strings in vocab; otherwise byte-fallback
            if token in self.vocab:
                if not debug:
                    return [self.vocab[token]]
                else:
                    return [self.vocab[token]], False
            else:
                if not debug:
                    return utf8_bytes_ids(self, token)
                else:
                    return utf8_bytes_ids(self, token), True
        elif self.paradigm == "morf":
            morphs = self.morf_model.viterbi_segment(token)[0]
            out: List[int] = []
            for m in morphs:
                if m in self.vocab:
                    out.append(self.vocab[m])
                else:
                    out.extend(utf8_bytes_ids(self, m))  # byte fallback
            return out
        return []

    def tokenize_with_eos(self, text):
        tokenized = self.tokenize(text)
        tokenized.extend([self.vocab.get("<EOS>", 0)])
        return tokenized

    def tokenize_corpus(self, path: str):
        """
        Stream-tokenize all text files in `path` (skips files with "tokenization" in the name),
        produce fixed-size binary shards in DATA_CACHE_DIR. Single-process; files are read
        line-by-line to avoid loading the entire corpus into memory.
        """
        DATA_CACHE_DIR = f"data/{self.language}_{self.paradigm}_CORPUS"
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)

        assert not self.trainable, "This needs to be trained, or load a tokenizer with a vocab"

        shard_size = 10 ** 8
        shard_index = 0

        # preallocate shard buffer
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

        # helper to start a new progress bar for the current shard
        def _new_progress(shard_idx):
            return tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_idx}")

        progress_bar = _new_progress(shard_index)

        try:
            # iterate files in given path (streaming)
            for filename in os.listdir(path):
                if "tokenization" in filename:
                    continue
                filepath = os.path.join(path, filename)
                if not os.path.isfile(filepath):
                    continue

                with open(filepath, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        tokens = self.tokenize_with_eos(line)

                        tokens = _flatten_to_ints(tokens)
                        idx = 0
                        remaining = len(tokens)

                        # Fill shards, possibly writing multiple shards if tokens exceed remainder
                        while remaining > 0:
                            space = shard_size - token_count
                            if space == 0:
                                # shard full: write and open new one
                                split = "val" if shard_index == 0 else "train"
                                outname = os.path.join(
                                    DATA_CACHE_DIR,
                                    f"{self.language}_{self.paradigm}_{split}_{shard_index:06d}.bin"
                                )
                                self.write_datafile(outname, all_tokens_np)
                                shard_index += 1
                                progress_bar.close()
                                progress_bar = _new_progress(shard_index)
                                token_count = 0
                                space = shard_size

                            take = min(space, remaining)
                            # place tokens[idx: idx + take] into buffer
                            all_tokens_np[token_count:token_count + take] = tokens[idx: idx + take]
                            token_count += take
                            idx += take
                            remaining -= take
                            progress_bar.update(take)
        finally:
            progress_bar.close()

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR,
                f"{self.language}_{self.paradigm}_{split}_{shard_index:06d}.bin"
            )
            # write only the portion filled
            self.write_datafile(filename, all_tokens_np[:token_count])

    def write_datafile(self, filename, toks):
        """
        Saves token data as a .bin file, for reading in C.
        - First comes a header with 256 int32s
        - The tokens follow, each as a uint16
        """
        assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
        # construct the header
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520 # magic
        header[1] = 1 # version
        header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
        # construct the tokens numpy array, if not already
        if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
            # validate that no token exceeds a uint16
            maxtok = 2**16
            assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
            toks_np = np.array(toks, dtype=np.uint16)
        else:
            toks_np = toks
        # write to file
        print(f"writing {len(toks):,} tokens to {filename}")
        with open(filename, "wb") as f:
            f.write(header.tobytes())
            f.write(toks_np.tobytes())
# endregion Tokenizer