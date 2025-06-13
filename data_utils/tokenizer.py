import os
import json
import re
import regex as re
from typing import Iterable, Iterator
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        super().__init__()
        self.vocab = vocab
        self.decoder = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self.bpe_ranks = dict(zip(self.merges, range(len(self.merges))))
        self.cache = {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls,
                   vocab_filepath: str | os.PathLike,
                   merges_filepath: str | os.PathLike,
                   special_tokens: list[str] | None = None):
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _load_vocab(filepath: str) -> dict[int, bytes]:
        with open(filepath) as vocab_f:
            vocab_raw = json.load(vocab_f)
        vocab = {}
        for k, v in vocab_raw.items():
            # assignment1: key must be int, value must be bytes
            try:
                key = int(k)
            except Exception:
                continue
            if isinstance(v, str):
                vocab[key] = v.encode('utf-8')
            else:
                vocab[key] = bytes(v)
        return vocab

    @staticmethod
    def _load_merges(filepath: str) -> list[tuple[bytes, bytes]]:
        merges = []
        with open(filepath, 'r', encoding='utf-8') as merges_f:
            for line in merges_f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pair = line.split()
                if len(pair) == 2:
                    merges.append((pair[0].encode('utf-8'), pair[1].encode('utf-8')))
        return merges

    def text_encode(self, text):
        vocab_reversed = {token: id_ for id_, token in self.vocab.items()}
        bpe_tokens_id = []
        for token_part in re.findall(self.pat, text):
            # Convert each character to its byte representation
            tokens = []
            for char in token_part:
                char_bytes = char.encode('utf-8')
                for byte in char_bytes:
                    tokens.append(bytes([byte]))
            
            # Apply BPE encoding
            bpe_tokens = self.bpe(tokens)
            
            # Convert BPE tokens to IDs
            for token in bpe_tokens:
                if token in vocab_reversed:
                    bpe_tokens_id.append(vocab_reversed[token])
                else:
                    # If token not in vocab, try to encode each byte
                    for byte in token:
                        byte_token = bytes([byte])
                        if byte_token in vocab_reversed:
                            bpe_tokens_id.append(vocab_reversed[byte_token])
        return bpe_tokens_id

    def bpe(self, tokens):
        if not tokens:
            return tokens
            
        reconstructed_text = b"".join(tokens).decode('utf-8')
        if reconstructed_text in self.cache:
            return self.cache[reconstructed_text]
            
        pairs = self._get_pairs(tokens)
        if not pairs:
            return tokens
            
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
                    
            tokens = new_tokens
            if len(tokens) == 1:
                break
            pairs = self._get_pairs(tokens)
            
        self.cache[reconstructed_text] = tokens
        return tokens

    def _get_pairs(self, word):
        """Get all adjacent pairs of characters in the word.
        Args:
            word: A tuple or list of bytes representing the word.
        Returns:
            A set of tuples, each containing two adjacent bytes.
        """
        pairs = set()
        if len(word) < 2:
            return pairs
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def encode(self, text):
        vocab = self.vocab
        special_tokens = self.special_tokens
        if special_tokens:
            vocab = add_special_tokens_to_vocab(vocab, special_tokens)
        vocab_reversed = {token: id_ for id_, token in vocab.items()}

        token_ids = []
        text_parts_mark = self._split_text_special_tokens(text, special_tokens)
        for text, is_special in text_parts_mark:
            if is_special:
                token_ids.append(vocab_reversed[text.encode('utf-8')])
            else:
                tokens = [bytes([b]) for b in text.encode('utf-8')]
                if len(tokens) == 1:
                    token_ids.append(vocab_reversed[tokens[0]])
                else:
                    if len(tokens) == 1:
                        token_ids.append(vocab_reversed[tokens[0]])
                    else:
                        token_ids.extend(self.text_encode(text))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        vocab = self.vocab
        special_tokens = self.special_tokens
        if special_tokens:
            vocab = add_special_tokens_to_vocab(vocab, special_tokens)
        res = []
        current_bytes = []
        for each_id in ids:
            token = vocab[each_id]
            current_bytes.append(token)
            try:
                char = b''.join(current_bytes).decode('utf-8')
                res.append(char)
                current_bytes = []
            except UnicodeDecodeError:
                pass
        if current_bytes:
            res.append(b''.join(current_bytes).decode('utf-8', errors='replace'))
        return ''.join(res)

    @staticmethod
    def _split_text_special_tokens(text, special_tokens):
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(token) for token in special_tokens)
            parts = []
            last_end = 0
            for match in re.finditer(special_pattern, text):
                start, end = match.span()
                if last_end < start:
                    parts.append((text[last_end:start], False))
                parts.append((text[start:end], True))
                last_end = end
            if last_end < len(text):
                parts.append((text[last_end:], False))
            return parts
        else:
            return [(text, False)]

    def _bpe(self, text: str) -> str:
        """
        Apply BPE merges to a text string (for test compatibility)
        
        Args:
            text: Input text string
            
        Returns:
            String representation of BPE tokens (space-separated)
        """
        if text in self.cache:
            return self.cache[text]

        if not text:
            self.cache[text] = text
            return text

        tokens = [bytes([b]) for b in text.encode('utf-8')]
        result_tokens = self.bpe(tokens)
        
        # Convert back to space-separated string
        result = ' '.join(token.decode('utf-8', errors='replace') for token in result_tokens)
        self.cache[text] = result
        return result

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """简化的BPE训练函数用于测试"""
    merges = []
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 简单的预处理
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = re.findall(PAT, content)
    
    # 构建基础vocab (256 bytes + special tokens)
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1
    
    # 计算字符频率
    char_counts = Counter()
    for token in tokens:
        for char in token:
            char_counts[char] += 1
    
    # 简单的合并策略：合并最常见的字符对
    num_merges = vocab_size - len(vocab)
    for _ in range(min(num_merges, 10)):  # 限制合并次数
        # 找到最常见的相邻字符对
        pair_counts = Counter()
        for token in tokens:
            for i in range(len(token) - 1):
                pair = (token[i].encode('utf-8'), token[i + 1].encode('utf-8'))
                pair_counts[pair] += 1
        
        if not pair_counts:
            break
            
        most_common_pair = pair_counts.most_common(1)[0][0]
        merges.append(most_common_pair)
        
        # 简单的token更新（这里简化处理）
        new_token = most_common_pair[0] + most_common_pair[1]
        vocab[len(vocab)] = new_token
    
    return vocab, merges

def add_special_tokens_to_vocab(vocab: dict[int, bytes], special_tokens: list[str]) -> dict[int, bytes]:
    if not special_tokens:
        return vocab
    updated_vocab = vocab.copy()
    max_id = max(updated_vocab.keys(), default=-1)
    existing_values = set(updated_vocab.values())
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        if token_bytes not in existing_values:
            max_id += 1
            updated_vocab[max_id] = token_bytes
            existing_values.add(token_bytes)
    return updated_vocab

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# === 以下为BPE训练相关函数 ===
def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
):
    merges = []
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()

    input_text = content
    vocab_size = vocab_size - 256 - len(special_tokens)

    # 1. Remove special tokens
    pattern = '|'.join(re.escape(token) for token in special_tokens)
    input_text = re.sub(pattern, '', input_text)

    # 2. Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    input_text = re.findall(PAT, input_text)

    input_text = Counter(input_text)
    token_counts = Counter(input_text)

    token_split_counts, token_split_tuple_map = split_tokens_with_end_marker(token_counts)
    in_token_adjacent_char_counts = count_adjacent_char_pairs(token_counts)
    pair_counts = count_in_token_pairs(in_token_adjacent_char_counts)
    pair_appear_index = get_pair_appear_index(token_counts)

    while vocab_size > 0:
        most_frequent_pair = get_most_freq_pair(pair_counts)
        merges.append(most_frequent_pair)
        vocab_size -= 1
        target_tokens = dict(pair_appear_index[most_frequent_pair])

        token_split_counts, token_split_tuple_map, in_token_adjacent_char_counts, pair_appear_index = merge_update(
            token_counts, target_tokens, token_split_tuple_map, token_split_counts, in_token_adjacent_char_counts,
            pair_appear_index, most_frequent_pair)

        pair_counts = count_in_token_pairs(in_token_adjacent_char_counts)

    vocab = {}
    merges = [(s1.encode('utf-8'), s2.encode('utf-8')) for s1, s2 in merges]
    for merge in merges:
        new_token = b''.join(merge)
        vocab[len(vocab)] = new_token
    return (vocab, merges)

def split_tokens_with_end_marker(token_counts):
    """
        Processes a dictionary of token counts and splits tokens into tuples.
        If a token ends with '</w>', it is treated as the end of a word and handled accordingly.

        Args:
            token_counts (dict): A dictionary where keys are tokens (str) and values are their counts (int).

        Returns:
            tuple: A tuple containing two dictionaries:
                - tuple_token_counts (dict): Maps tuples of characters (or tokens) to their counts.
                - full_token_tuple_token (dict): Maps original tokens to their tuple representations.

        Example:
            Input: {'hello</w>': 5, 'world': 3}
            Output: (
                {('h', 'e', 'l', 'l', 'o', '</w>'): 5, ('w', 'o', 'r', 'l', 'd'): 3},
                {'hello</w>': ('h', 'e', 'l', 'l', 'o', '</w>'), 'world': ('w', 'o', 'r', 'l', 'd')}
            )
        """
    token_split_counts = {}
    token_split_tuple_map = {}
    for token, count in token_counts.items():
        if token.endswith('</w>'):
            word_part = token[:-4]
            token_split_counts[tuple(word_part) + ('</w>',)] = count
            token_split_tuple_map[token] = tuple(word_part) + ('</w>',)
        else:
            token_split_counts[tuple(token)] = count
            token_split_tuple_map[token] = tuple(token)
    return token_split_counts, token_split_tuple_map

def count_adjacent_char_pairs(token_counts):
    """
    Counts adjacent character pairs and their frequencies in all tokens, avoiding splitting '</w>'.
    The frequency of each pair is weighted by the token's count.

    Args:
        token_counts (dict): A dictionary where keys are tokens (str) and values are their counts (int).

    Returns:
        dict: A dictionary mapping each token to another dictionary that maps character pairs to their weighted frequencies.

    Example:
        Input: {'hello</w>': 5, 'world': 3}
        Output: {
            'hello</w>': {('h', 'e'): 5, ('e', 'l'): 5, ('l', 'l'): 5, ('l', 'o'): 5, ('o', '</w>'): 5},
            'world': {('w', 'o'): 3, ('o', 'r'): 3, ('r', 'l'): 3, ('l', 'd'): 3}
        }
    """
    in_token_adjacent_char_counts = Counter()

    for token, count in token_counts.items():
        pair_counts = Counter()
        if len(token) == 1:
            continue
        elif token.endswith('</w>'):
            # 特殊处理：保留 '</w>'，将其分离为独立部分
            word_part = token[:-4]  # 去掉 '</w>'
            # 统计单词部分的字符对，并按 count 加权
            pair_counts.update(
                {pair: freq * count for pair, freq in Counter(zip(word_part[:-1], word_part[1:])).items()})
            # 将 '</w>' 添加为最后一个字符对，并按 count 加权
            if word_part:
                pair_counts[(word_part[-1], '</w>')] += count
        else:
            # 普通情况：直接统计字符对，并按 count 加权
            pair_counts.update({pair: freq * count for pair, freq in Counter(zip(token[:-1], token[1:])).items()})

        in_token_adjacent_char_counts[token] = pair_counts
    return in_token_adjacent_char_counts

def count_in_token_pairs(in_token_counts):
    """
    Aggregates all character pair counts from a nested dictionary.

    Args:
        in_token_counts (dict): A dictionary where keys are tokens (str) and values are dictionaries
                                mapping character pairs (tuples) to their counts (int).

    Returns:
        Counter: A Counter object aggregating all character pair counts across tokens.

    Example:
        Input: {
            'hello</w>': {('h', 'e'): 5, ('e', 'l'): 5, ('l', 'l'): 5, ('l', 'o'): 5, ('o', '</w>'): 5},
            'world': {('w', 'o'): 3, ('o', 'r'): 3, ('r', 'l'): 3, ('l', 'd'): 3}
        }
        Output: Counter({('h', 'e'): 5, ('e', 'l'): 5, ('l', 'l'): 5, ('l', 'o'): 5, ('o', '</w>'): 5, ('w', 'o'): 3, ('o', 'r'): 3, ('r', 'l'): 3, ('l', 'd'): 3})
    """
    res = Counter()
    for counter in in_token_counts.values():
        res.update(counter)
    return res

def get_most_freq_pair(pair_counts_dict):
    # Back to original simple strategy
    return max(
        pair_counts_dict,
        key=lambda pair: (pair_counts_dict[pair], pair)
    )

def get_pair_appear_index(token_counts):
    """
    Finds the positions of all adjacent character pairs in the given tokens.

    Args:
        token_counts (dict): A dictionary where keys are tokens (str) and values are their counts (int).

    Returns:
        defaultdict: A dictionary where keys are character pairs (tuples) and values are lists of tuples
                     indicating the token and position where the pair appears.

    Example:
        Input: {' you</w>': 1, 'u</w>': 3}
        Output: {('u', '</w>'): [('u</w>', 0), (' you</w>', 3)]...}
    """
    pair_index = defaultdict(list)
    for token, freq in token_counts.items():
        token_chars = []
        # 将特殊符号与普通字符区分
        i = 0
        while i < len(token):
            # 如果检测到 '<' 开头的特殊标记，则整体处理
            if token[i] == '<':
                j = token.find('>', i) + 1
                token_chars.append(token[i:j])
                i = j
            else:
                token_chars.append(token[i])
                i += 1

        # 记录相邻符号对
        for i in range(len(token_chars) - 1):
            pair = (token_chars[i], token_chars[i + 1])
            pair_index[pair].append((token, i))  # (token, 位置)
    return pair_index

def merge_update(token_counts, target_tokens, token_split_tuple_map, token_split_counts, in_token_adjacent_char_counts,
                 pair_appear_index, merge_pair):
    """
    Merges character pairs into a single token and updates related mappings.

    Args:
        target_tokens (dict): A dictionary mapping tokens to their positions where a merge should occur.
        token_split_tuple_map (dict): A dictionary mapping tokens to their tuple representations.
        token_split_counts (dict): A dictionary mapping token tuples to their counts.
        in_token_adjacent_char_counts (dict): A dictionary mapping tokens to their adjacent character pair counts.

    Returns:
        tuple: Updated versions of:
            - token_split_counts (dict): Updated counts for token tuples.
            - token_split_tuple_map (dict): Updated mappings of tokens to their tuple representations.
            - in_token_adjacent_char_counts (dict): Updated adjacent character pair counts for tokens.
            - pair_appear_index_new (defaultdict): A dictionary where keys are updated pairs and values are lists of tuples
                                                  indicating the token and position where the pair appears.

    Example:
        Input: target_tokens = {'abc': 1}, token_split_tuple_map = {'abc': ('a', 'b', 'c')},
               token_split_counts = {('a', 'b', 'c'): 5}, in_token_adjacent_char_counts = {'abc': Counter()}
        Output: Updated dictionaries reflecting merged tokens and adjusted counts.
    """

    for token, _ in target_tokens.items():
        count = token_counts[token]
        split_token = token_split_tuple_map[token]
        old_key = split_token

        new_key = merge_all_keys(old_key, merge_pair)

        token_split_counts[new_key] = token_split_counts.pop(old_key)

        token_split_tuple_map[token] = new_key

        adjacent_pairs = [(new_key[i], new_key[i + 1]) for i in range(len(new_key) - 1)]

        pair_appear_index = update_pair_appear_index(pair_appear_index, old_key, new_key, token)

        pair_counts = Counter(adjacent_pairs)
        pair_counts = Counter({key: pair_counts[key] * count for key in pair_counts})

        in_token_adjacent_char_counts[token] = pair_counts

    return token_split_counts, token_split_tuple_map, in_token_adjacent_char_counts, pair_appear_index

def update_pair_appear_index(pair_token_index, old_key, new_key, cur_token):
    """
    pair_token_index: {('u', '</w>'): [('u</w>', 0), (' you</w>', 3)]...}
    """

    old_adjacent_pairs = [(old_key[i], old_key[i + 1]) for i in range(len(old_key) - 1)]
    adjacent_pairs = [(new_key[i], new_key[i + 1]) for i in range(len(new_key) - 1)]
    to_remove_keys = set(old_adjacent_pairs) - set(adjacent_pairs)
    to_add_key = set(adjacent_pairs) - set(old_adjacent_pairs)

    if len(old_adjacent_pairs) == 1:

        key_to_remove = old_adjacent_pairs[0]
        word_to_remove = lambda x: cur_token in x[0]
        pair_token_index[key_to_remove] = [token_and_idx for token_and_idx in pair_token_index[key_to_remove] if
                                           not word_to_remove(token_and_idx)]

        if not pair_token_index[old_adjacent_pairs[0]]:
            del pair_token_index[old_adjacent_pairs[0]]
    else:

        for to_add_key_i in to_add_key:

            # 1. 增加新的组合
            for idx, pair in enumerate(adjacent_pairs):
                if pair == to_add_key_i:
                    merged_pair = ''.join(to_add_key_i)

                    indexes = [i for i in range(len(cur_token) - len(merged_pair) + 1) if
                               cur_token[i:i + len(merged_pair)] == merged_pair]

                    for i in indexes:
                        pair_token_index[pair].append((cur_token, i))
            for old_key_i in to_remove_keys:
                key_to_remove = old_key_i
                word_to_remove = lambda x: cur_token in x[0]
                pair_token_index[key_to_remove] = [token_and_idx for token_and_idx in pair_token_index[key_to_remove] if
                                                   not word_to_remove(token_and_idx)]
    return pair_token_index

def merge_all_keys(old_key, merge_pair):
    merge_pair_tuple = tuple(merge_pair)
    result = []
    i = 0
    while i < len(old_key):
        if i < len(old_key) - 1 and old_key[i:i + 2] == merge_pair_tuple:
            result.append(''.join(merge_pair_tuple))
            i += 2
        else:
            result.append(old_key[i])
            i += 1
    return tuple(result)