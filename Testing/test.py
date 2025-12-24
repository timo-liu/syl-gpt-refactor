from Definitions.Tokenizer import Tokenizer, TokenizerConfig
import argparse
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_config")
    parser.add_argument("bin")
    args = parser.parse_args()

    tokenizer_config = TokenizerConfig.load(args.tokenizer_config)
    tokenizer = Tokenizer(tokenizer_config)

    with open(args.bin, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
        # separators = (tokens == tokenizer.vocab.get("<EOS>", 0))
        separators = (tokens > (257 if tokenizer.paradigm == "syl" else 259)) & (
                    tokens <= (285 if tokenizer.paradigm == "syl" else 287))

        sep_idx = np.where(separators)[0]
        splits = np.split(tokens, sep_idx + 1)
    for s in splits[:10]:
        print(len(s)-3)
        print(" ".join(
            tokenizer.i2c[x].decode("utf-8", errors="ignore")
            if isinstance(tokenizer.i2c[x], bytes)
            else str(tokenizer.i2c[x])
            for x in s
        ))
        print(tokenizer.decode(s))