#!/usr/bin/env python3
"""
Adapter functions for testing educational transformer implementation
Following the exact pattern from original assignment 1 tests
"""
from __future__ import annotations

import os
from typing import IO, BinaryIO, Iterable, Optional, Type
from pathlib import Path

import numpy.typing as npt
import torch

import regex as re
from collections import Counter, defaultdict

from transformer_lib.feedforward import FeedForwardNetwork
from transformer_lib.normalization import RMSNorm
from transformer_lib.activations import GELU, Softmax
from transformer_lib.attention import ScaledDotProductAttention, MultiHeadSelfAttention
from transformer_lib.transformer_block import TransformerBlock
from transformer_lib.transformer_model import TransformerLanguageModel
from transformer_lib.optimizers import AdamW
from data_utils.data_loader import create_data_loader
from data_utils.tokenizer import BPETokenizer


def run_positionwise_feedforward(
        d_model: int,
        d_ff: int,
        weights: dict[str, torch.FloatTensor],
        in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a position-wise feedforward network, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are `w1.weight` and `w2.weight`.
            `w1` is the first linear transformation, and `w2` is the second
            linear transformation (eq. 2 of Vaswani et al., 2017).
            `w1.weight` is of shape (d_ff, d_model).
            `w2.weight` is of shape (d_model, d_ff).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your position-wise feedforward network
        with the provided `weights` on the provided `in_features`.
    """
    ffn = FeedForwardNetwork(d_model, d_ff)
    ffn.load_state_dict(weights)
    output = ffn(in_features)
    return output


def run_scaled_dot_product_attention(
        K: torch.FloatTensor,
        Q: torch.FloatTensor,
        V: torch.FloatTensor,
        mask: Optional[torch.BoolTensor] = None,
        pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float]
            Probability of dropout to be applied after
            the attention softmax operation.

    Returns:
        torch.FloatTensor with the output of running scaled dot product attention
        on the provided Q, K, V tensors, optionally using the provided
        `mask` and `pdrop`.
    """
    if pdrop is None:
        pdrop = 0.0
    
    attention = ScaledDotProductAttention(pdrop)
    output = attention(Q, K, V, mask)
    return output


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        weights: dict[str, torch.FloatTensor],
        in_features: torch.FloatTensor,
        pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given the weights of a multihead self-attention layer, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the model.
        num_heads: int
            Number of attention heads.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
        pdrop: Optional[float]
            Probability of dropout to be applied.

    Returns:
        torch.FloatTensor with the output of running your multihead self-attention
        with the provided `weights` on the provided `in_features`.
    """
    if pdrop is None:
        pdrop = 0.0
    
    attention = MultiHeadSelfAttention(d_model, num_heads, pdrop)
    attention.load_state_dict(weights)
    output = attention(in_features)
    return output


def run_rmsnorm(
        d_model: int,
        weights: dict[str, torch.FloatTensor],
        in_features: torch.FloatTensor,
        eps: float = 1e-5,
) -> torch.FloatTensor:
    """Given the weights of an RMSNorm layer, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the model.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The key of this dictionary is `weight`.
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
        eps: float
            Epsilon value for numerical stability.

    Returns:
        torch.FloatTensor with the output of running your RMSNorm
        with the provided `weights` on the provided `in_features`.
    """
    norm = RMSNorm(d_model, epsilon=eps)
    norm.load_state_dict(weights)
    output = norm(in_features)
    return output


def run_gelu(
        in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given input features, return the output of your GELU implementation.

    Args:
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your GELU
        on the provided `in_features`.
    """
    gelu = GELU()
    output = gelu(in_features)
    return output


def run_softmax(
        in_features: torch.FloatTensor,
        dim: int = -1,
) -> torch.FloatTensor:
    """Given input features, return the output of your Softmax implementation.

    Args:
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
        dim: int
            Dimension along which to apply softmax.

    Returns:
        torch.FloatTensor with the output of running your Softmax
        on the provided `in_features`.
    """
    softmax = Softmax(dim)
    output = softmax(in_features)
    return output


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
        weights: dict[str, torch.FloatTensor],
        in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a transformer block, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the model.
        num_heads: int
            Number of attention heads.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        attn_pdrop: float
            Attention dropout probability.
        residual_pdrop: float
            Residual dropout probability.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your transformer block
        with the provided `weights` on the provided `in_features`.
    """
    block = TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
    block.load_state_dict(weights)
    output = block(in_features)
    return output


def run_transformer_lm(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
        weights: dict[str, torch.FloatTensor],
        in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    """Given the weights of a transformer language model, return
    the output of your implementation with these weights.

    Args:
        vocab_size: int
            Size of the vocabulary.
        context_length: int
            Maximum sequence length the model can handle.
        d_model: int
            Dimensionality of the model.
        num_layers: int
            Number of transformer layers.
        num_heads: int
            Number of attention heads.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        attn_pdrop: float
            Attention dropout probability.
        residual_pdrop: float
            Residual dropout probability.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
        in_indices: torch.LongTensor
            Token indices to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your transformer language model
        with the provided `weights` on the provided `in_indices`.
    """
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_dropout=attn_pdrop,
        residual_dropout=residual_pdrop
    )
    model.load_state_dict(weights)
    model.eval()
    output = model(in_indices)
    return output


def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    """Return your AdamW optimizer class.

    Returns:
        Your AdamW optimizer class.
    """
    return AdamW


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    merges = []
    with input_path.open("r", encoding="utf-8") as file:
        content = file.read()

    input_text = content

    num_merges = vocab_size - 256 - len(special_tokens)

    # 1. Remove special tokens
    pattern = '|'.join(re.escape(token) for token in special_tokens)
    input_text = re.sub(pattern, '', input_text)

    # 2. Pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    input_text = re.findall(PAT, input_text)

    # 3. Compute BPE merges
    input_text = Counter(input_text)
    token_counts = Counter(input_text)

    token_split_counts, token_split_tuple_map = split_tokens_with_end_marker(token_counts)
    in_token_adjacent_char_counts = count_adjacent_char_pairs(token_counts)
    pair_counts = count_in_token_pairs(in_token_adjacent_char_counts)
    pair_appear_index = get_pair_appear_index(token_counts)

    while num_merges > 0:
        most_frequent_pair = get_most_freq_pair(pair_counts)
        merges.append(most_frequent_pair)
        num_merges -= 1
        target_tokens = dict(pair_appear_index[most_frequent_pair])

        token_split_counts, token_split_tuple_map, in_token_adjacent_char_counts, pair_appear_index = merge_update(
            token_counts, target_tokens, token_split_tuple_map, token_split_counts, in_token_adjacent_char_counts,
            pair_appear_index, most_frequent_pair)

        pair_counts = count_in_token_pairs(in_token_adjacent_char_counts)

    # 4. Build complete vocabulary
    vocab = {}
    
    # Add special tokens first
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    
    # Add 256 base bytes 
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    
    # Convert merges to bytes and add merged tokens
    merges_bytes = [(s1.encode('utf-8'), s2.encode('utf-8')) for s1, s2 in merges]
    for merge in merges_bytes:
        new_token = b''.join(merge)
        vocab[len(vocab)] = new_token

    return (vocab, merges_bytes)


def get_bpe_tokenizer_cls() -> Type[BPETokenizer]:
    """Return your BPE tokenizer class.

    Returns:
        Your BPE tokenizer class.
    """
    return BPETokenizer


def get_tokenizer(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
) -> BPETokenizer:
    """Create a BPE tokenizer with the given vocabulary, merges, and special tokens.

    Args:
        vocab: Dictionary mapping token IDs to byte strings.
        merges: List of merge rules as (bytes1, bytes2) tuples.
        special_tokens: Optional list of special tokens.

    Returns:
        A BPE tokenizer instance.
    """
    return BPETokenizer(vocab, merges, special_tokens)


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

def get_most_freq_pair(pair_counts_dict):
    # Back to original simple strategy
    return max(
        pair_counts_dict,
        key=lambda pair: (pair_counts_dict[pair], pair)
    )

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

def merge_all_keys(old_key, merge_pair):
    merge_pair_tuple = tuple(merge_pair)
    result = []
    i = 0
    while i < len(old_key):
        # 检查当前位置是否匹配 merge_pair
        if i < len(old_key) - 1 and old_key[i:i + 2] == merge_pair_tuple:
            # 合并匹配的 pair
            result.append(''.join(merge_pair_tuple))
            i += 2  # 跳过已合并的部分
        else:
            # 否则保留当前元素
            result.append(old_key[i])
            i += 1
    return tuple(result)

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

            # 2. 删除不用的旧的
            for old_key_i in to_remove_keys:
                key_to_remove = old_key_i
                word_to_remove = lambda x: cur_token in x[0]

                pair_token_index[key_to_remove] = [token_and_idx for token_and_idx in pair_token_index[key_to_remove] if
                                                   not word_to_remove(token_and_idx)]

    return pair_token_index
