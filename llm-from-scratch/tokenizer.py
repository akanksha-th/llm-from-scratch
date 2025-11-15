class BPETokenizer:
    """ A minimal Byte Pair Encoding Tokenizer implemented from scratch"""
    def __init__(self, content_file: str):
        self.vocab_size = 256
        self.merges = []
        self.merge_lookup = {}
        self.reverse_lookup = {}
        with open(content_file, "r", encoding="utf-8") as f:
            self._content = f.read()

    def __pair_count(self, corpus):
        """
        Count the frequency of adjacent token pairs across all sequences in the corpus.
        Returns: dict[(int, int)] = frequency
        """
        counts = {}
        for seq in corpus:
            for i in range(len(seq)-1):
                pair = (seq[i], seq[i+1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def __merge_pairs(self, corpus, pair, new_token):
        """
        Merge a given pair of sequences of all the corpus.
        Returns: new corpus with merged pairs.
        """
        new_corpus = []
        for seq in corpus:
            i = 0
            new_seq = []
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_corpus.append(new_seq)
        return new_corpus

    def train(self, num_merges = 1000):
        """
        Train the BPE tokenizer on the given corpus.
        """
        corpus = [list(self._content.encode("utf-8"))]

        # Count Pairs
        for _ in range(num_merges):
            counts = self.__pair_count(corpus)
            if not counts:
                break

            pair = max(counts, key=counts.get)
            new_token = self.vocab_size

            self.merges.append((pair, new_token))
            self.merge_lookup[pair] = new_token
            self.reverse_lookup[new_token] = pair

            corpus = self.__merge_pairs(corpus, pair, new_token)
            self.vocab_size += 1

        print(f"Training Complete: {len(self.merges)} merges learned.")

    def encode(self, text: str):
        """
        Encode new text using learned BPE merges.
        """
        seq = list(text.encode("utf-8"))
        for (a, b), new_tok in self.merges:
            i = 0
            new_seq = []
            while i < len(seq):
                if i < len(seq)-1 and (seq[i], seq[i+1]) == (a,b):
                    new_seq.append(new_tok)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        return seq

    def _expand_token(self, token):
        if token in self.reverse_lookup:
            a, b = self. reverse_lookup[token]
            return self._expand_token(a) + self._expand_token(b)
        else:
            return [token]

    def decode(self, tokens):
        """
        Decode BPE tokens back to text.
        """
        expanded = []
        for token in tokens:
            expanded.extend(self._expand_token(token))
        return bytes(expanded).decode("utf-8")

tokenizer = BPETokenizer("C:/Users/aktkr/llm/story.txt")
tokenizer.train()

encoded = tokenizer.encode("the quick brown fox jumps over the lazy dog")
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)
