import pytest
import torch
import numpy as np
import tempfile
import os
from data_utils.data_loader import create_data_loader, load_dataset
from data_utils.tokenizer import BPETokenizer, run_train_bpe


class TestDataLoader:
    """Test cases for data loader functionality"""
    
    def test_data_loader_basic_functionality(self, device):
        """Test basic data loader functionality"""
        # Create sample data
        data = torch.arange(100)
        batch_size = 4
        context_length = 8
        
        inputs, targets = create_data_loader(data, batch_size, context_length, device)
        
        assert inputs.shape == (batch_size, context_length)
        assert targets.shape == (batch_size, context_length)
        assert inputs.device.type == device.type
        assert targets.device.type == device.type
    
    def test_data_loader_target_shift(self, device):
        """Test that targets are correctly shifted by one position"""
        # Create sequential data for easy verification
        data = torch.arange(20)
        batch_size = 2
        context_length = 5
        
        # Set seed for reproducible random indices
        torch.manual_seed(42)
        inputs, targets = create_data_loader(data, batch_size, context_length, device)
        
        # Targets should be inputs shifted by 1
        for i in range(batch_size):
            for j in range(context_length):
                expected_target = inputs[i, j] + 1
                assert targets[i, j] == expected_target
    
    def test_data_loader_input_validation(self, device):
        """Test data loader input validation"""
        batch_size = 4
        context_length = 8
        
        # Test with non-tensor input
        with pytest.raises(TypeError, match="Input data must be a torch.Tensor"):
            create_data_loader([1, 2, 3, 4], batch_size, context_length, device)
        
        # Test with insufficient data
        small_data = torch.arange(10)  # Too small for batch_size * context_length
        with pytest.raises(ValueError, match="Dataset is too small"):
            create_data_loader(small_data, batch_size, context_length, device)
    
    def test_data_loader_random_sampling(self, device):
        """Test that data loader produces different batches with random sampling"""
        data = torch.arange(1000)
        batch_size = 4
        context_length = 8
        
        # Generate multiple batches
        batches = []
        for _ in range(5):
            inputs, targets = create_data_loader(data, batch_size, context_length, device)
            batches.append((inputs, targets))
        
        # Check that not all batches are identical (very unlikely with random sampling)
        all_same = all(
            torch.allclose(batches[0][0], batch[0]) and torch.allclose(batches[0][1], batch[1])
            for batch in batches[1:]
        )
        assert not all_same, "All batches are identical, random sampling may not be working"
    
    def test_data_loader_edge_cases(self, device):
        """Test data loader with edge cases"""
        # Minimum valid data size
        data = torch.arange(32)  # Exactly enough for batch_size=4, context_length=8
        batch_size = 4
        context_length = 8
        
        inputs, targets = create_data_loader(data, batch_size, context_length, device)
        assert inputs.shape == (batch_size, context_length)
        assert targets.shape == (batch_size, context_length)
        
        # Single sequence
        batch_size = 1
        context_length = 1
        data = torch.arange(10)
        
        inputs, targets = create_data_loader(data, batch_size, context_length, device)
        assert inputs.shape == (1, 1)
        assert targets.shape == (1, 1)
    
    def test_data_loader_data_types(self, device):
        """Test data loader with different data types"""
        batch_size = 2
        context_length = 4
        
        # Test with different integer types
        for dtype in [torch.int32, torch.int64, torch.long]:
            data = torch.arange(50, dtype=dtype)
            inputs, targets = create_data_loader(data, batch_size, context_length, device)
            assert inputs.dtype == dtype
            assert targets.dtype == dtype
    
    def test_data_loader_large_data(self, device):
        """Test data loader with larger datasets"""
        # Simulate larger dataset
        data = torch.randint(0, 1000, (10000,))
        batch_size = 16
        context_length = 32
        
        inputs, targets = create_data_loader(data, batch_size, context_length, device)
        
        assert inputs.shape == (batch_size, context_length)
        assert targets.shape == (batch_size, context_length)
        assert not torch.any(torch.isnan(inputs.float()))
        assert not torch.any(torch.isnan(targets.float()))


class TestLoadDataset:
    """Test cases for dataset loading functionality"""
    
    def test_load_dataset_basic(self):
        """Test basic dataset loading from numpy file"""
        # Create temporary numpy file
        test_data = np.arange(100, dtype=np.int32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            np.save(tmp_file.name, test_data)
            tmp_path = tmp_file.name
        
        try:
            # Load dataset
            loaded_data = load_dataset(tmp_path)
            
            assert isinstance(loaded_data, np.memmap)
            assert loaded_data.dtype == np.int32
            assert len(loaded_data) == 100
            np.testing.assert_array_equal(loaded_data, test_data)
        finally:
            os.unlink(tmp_path)
    
    def test_load_dataset_different_dtypes(self):
        """Test loading datasets with different data types"""
        dtypes_to_test = [np.int32, np.int64, np.float32, np.float64]
        
        for dtype in dtypes_to_test:
            test_data = np.arange(50, dtype=dtype)
            
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
                np.save(tmp_file.name, test_data)
                tmp_path = tmp_file.name
            
            try:
                loaded_data = load_dataset(tmp_path, dtype=dtype)
                assert loaded_data.dtype == dtype
                np.testing.assert_array_equal(loaded_data, test_data)
            finally:
                os.unlink(tmp_path)
    
    def test_load_dataset_memory_mapping(self):
        """Test that loaded dataset is memory-mapped"""
        test_data = np.arange(1000, dtype=np.int32)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            np.save(tmp_file.name, test_data)
            tmp_path = tmp_file.name
        
        try:
            loaded_data = load_dataset(tmp_path)
            
            # Check that it's a memory-mapped array
            assert isinstance(loaded_data, np.memmap)
            
            # Memory-mapped arrays should have mode 'r' (read-only)
            assert loaded_data.mode == 'r'
        finally:
            os.unlink(tmp_path)


class TestBPETokenizer:
    """Test cases for BPE Tokenizer"""
    
    @pytest.fixture
    def sample_vocab(self):
        """Sample vocabulary for testing"""
        return {
            0: b'h',
            1: b'e', 
            2: b'l',
            3: b'o',
            4: b' ',
            5: b'w',
            6: b'r',
            7: b'd',
            8: b'he',
            9: b'el',
            10: b'll',
            11: b'lo',
            12: b'llo',
            13: b'hello'
        }
    
    @pytest.fixture
    def sample_merges(self):
        """Sample merge rules for testing"""
        return [
            (b'h', b'e'),    # -> 'he'
            (b'e', b'l'),    # -> 'el' 
            (b'l', b'l'),    # -> 'll'
            (b'l', b'o'),    # -> 'lo'
            (b'll', b'o'),   # -> 'llo'
            (b'he', b'llo'), # -> 'hello'
        ]
    
    def test_tokenizer_initialization(self, sample_vocab, sample_merges):
        """Test BPE tokenizer initializes correctly"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        assert tokenizer.vocab == sample_vocab
        assert tokenizer.merges == sample_merges
        assert len(tokenizer.decoder) == len(sample_vocab)
        assert len(tokenizer.bpe_ranks) == len(sample_merges)
        assert tokenizer.cache == {}
    
    def test_tokenizer_from_files(self):
        """Test loading tokenizer from files"""
        # Create temporary vocab file
        vocab_data = {"0": "h", "1": "e", "2": "l", "3": "o"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as vocab_file:
            import json
            json.dump(vocab_data, vocab_file)
            vocab_path = vocab_file.name
        
        # Create temporary merges file
        merges_data = "h e\ne l\nl l\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as merges_file:
            merges_file.write(merges_data)
            merges_path = merges_file.name
        
        try:
            tokenizer = BPETokenizer.from_files(vocab_path, merges_path)
            assert isinstance(tokenizer, BPETokenizer)
            assert len(tokenizer.vocab) == 4
            assert len(tokenizer.merges) == 3
        finally:
            os.unlink(vocab_path)
            os.unlink(merges_path)
    
    def test_tokenizer_get_pairs(self, sample_vocab, sample_merges):
        """Test getting pairs from word"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        word = (b'h', b'e', b'l', b'l', b'o')
        pairs = tokenizer._get_pairs(word)
        
        expected_pairs = {(b'h', b'e'), (b'e', b'l'), (b'l', b'l'), (b'l', b'o')}
        assert pairs == expected_pairs
    
    def test_tokenizer_bpe_simple(self, sample_vocab, sample_merges):
        """Test BPE encoding for simple cases"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        # Test simple word that should be merged
        result = tokenizer._bpe("hello")
        
        # Should apply merges according to the rules
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_tokenizer_caching(self, sample_vocab, sample_merges):
        """Test that tokenizer caches results"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        # First call should cache the result
        result1 = tokenizer._bpe("test")
        assert "test" in tokenizer.cache
        
        # Second call should use cache
        result2 = tokenizer._bpe("test")
        assert result1 == result2
    
    def test_tokenizer_special_tokens(self, sample_vocab, sample_merges):
        """Test tokenizer with special tokens"""
        special_tokens = ["<|endoftext|>", "<|pad|>"]
        tokenizer = BPETokenizer(sample_vocab, sample_merges, special_tokens)
        
        assert tokenizer.special_tokens == special_tokens
    
    def test_tokenizer_regex_pattern(self, sample_vocab, sample_merges):
        """Test tokenizer regex pattern compilation"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        # Pattern should be compiled
        assert tokenizer.pat is not None
        
        # Test pattern matching
        test_text = "hello world"
        matches = tokenizer.pat.findall(test_text)
        assert len(matches) > 0
    
    def test_tokenizer_empty_input(self, sample_vocab, sample_merges):
        """Test tokenizer with empty or invalid inputs"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        # Test empty string
        result = tokenizer._bpe("")
        assert result == ""
        
        # Test single character
        result = tokenizer._bpe("a")
        assert isinstance(result, str)
    
    def test_tokenizer_consistency(self, sample_vocab, sample_merges):
        """Test tokenizer produces consistent results"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        test_text = "hello"
        result1 = tokenizer._bpe(test_text)
        result2 = tokenizer._bpe(test_text)
        
        assert result1 == result2
    
    def test_tokenizer_merges_ranking(self, sample_vocab, sample_merges):
        """Test that merges are applied in correct priority order"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        # Check that bpe_ranks contains all merges
        assert len(tokenizer.bpe_ranks) == len(sample_merges)
        
        # Check that rankings are sequential
        ranks = list(tokenizer.bpe_ranks.values())
        assert ranks == list(range(len(sample_merges)))
    
    def test_tokenizer_decoder_consistency(self, sample_vocab, sample_merges):
        """Test that decoder is inverse of vocab"""
        tokenizer = BPETokenizer(sample_vocab, sample_merges)
        
        # Check that decoder is correctly constructed
        for token_id, token_bytes in sample_vocab.items():
            assert tokenizer.decoder[token_bytes] == token_id
        
        # Check roundtrip consistency
        for token_id, token_bytes in sample_vocab.items():
            decoded_id = tokenizer.decoder[token_bytes]
            assert decoded_id == token_id
    
    def test_tokenizer_train_bpe(self):
        """Test BPE training functionality"""
        input_text = "This is a sample text for testing the BPE tokenizer implementation."
        vocab_size = 300
        special_tokens = ["<s>", "</s>"]
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
            tmp.write(input_text)
            tmp.flush()
            vocab, merges = run_train_bpe(tmp.name, vocab_size, special_tokens)
        assert isinstance(vocab, dict)
        assert isinstance(merges, list)
        assert len(vocab) <= vocab_size
        assert len(merges) <= vocab_size - 256 - len(special_tokens)