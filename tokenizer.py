from extra import get_stats, merge
import regex as re

class Tokenizer():
    def __init__(self) -> None:
        self.merges = {}
        self.vocab = {}
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_re = re.compile(self.GPT4_SPLIT_PATTERN)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        toks_re = re.findall(self.compiled_re, text)
        stats = {}
        tokens = [list(t.encode('utf-8')) for t in toks_re]
        # print(tokens)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            for ch in tokens:
                get_stats(ch, stats)
            max_stat = max(stats, key=stats.get)
            idx = i + 256
            tokens = [merge(ch, max_stat, idx) for ch in tokens]
            vocab[idx] = vocab[max_stat[0]] + vocab[max_stat[1]]
            merges[max_stat] = idx
        self.merges = merges
        self.vocab = vocab

    def decode(self,ids):
        raw_byt = b"".join(self.vocab[i] for i in ids)
        text = raw_byt.decode('utf-8', errors='replace')
        return text

    def encode(self, text):
        toks_re = re.findall(self.compiled_re, text)
        ids=[]
        for t in toks_re:
            toks = list(t.encode('utf-8'))
            while len(toks) > 2:
                stats = get_stats(toks)
                pair = min(stats, key=lambda p: self.merges.get(p, -float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                toks = merge(toks, pair, idx)
            ids.extend(toks)
        return ids

        
    
