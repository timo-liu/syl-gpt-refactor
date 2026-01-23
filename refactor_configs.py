import os
from Definitions.Tokenizer import  TokenizerConfig, Tokenizer
import sentencepiece as spm
import sentencepiece_model_pb2 as sp_pb2


IPA = [
    "p","b","t","d","k","g",
    "f","v","θ","ð","s","z","ʃ","ʒ","h",
    "t","ʃ","d","ʒ",
    "m","n","ŋ",
    "l","ɹ","j","w",
    "i","ɪ","e","ɛ","æ",
    "ɑ","ɔ","o","ʊ","u",
    "ə","ʌ","ɜ","ɚ",
    "a","ɐ",
    "ː","ˈ","ˌ","."
]

for file in os.listdir("Configs"):
    j = 0
    if "syl" in file:
        tokenizer = Tokenizer(TokenizerConfig(os.path.join("Configs", file)))
        for i in IPA:
            if i not in tokenizer.vocab:
                j += 1
                tokenizer.vocab[i] = len(tokenizer.vocab) + 1
        print(f"Added: {j}")
        tokenizer.save_tokenizer(os.path.join("Configs", f"ipad_{file}"))
    elif ".model" in file: