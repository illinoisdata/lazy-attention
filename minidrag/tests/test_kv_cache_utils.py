import pytest
from unittest.mock import Mock, patch
from vllm.v1.core.kv_cache_utils import hash_request_tokens_no_prefix

@pytest.fixture
def mock_request():
    """Fixture to create a mock request object."""
    request = Mock()
    request.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return request

@pytest.fixture
def mock_hash_function():
    """Fixture to create a mock hash function."""
    return Mock(return_value="mock_hash")

def test_hash_request_tokens_no_prefix_basic(mock_hash_function, mock_request):
    """Test basic functionality with no extra keys."""
    block_size = 3
    expected_blocks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    with patch("vllm.v1.core.kv_cache_utils.need_extra_keys", return_value=False):
        with patch("vllm.v1.core.kv_cache_utils.hash_block_tokens") as mock_hash_block_tokens:
            mock_hash_block_tokens.side_effect = lambda hf, pb, tokens, ek: f"hash_{tokens}"

            result = hash_request_tokens_no_prefix(mock_hash_function, block_size, mock_request)

            # Verify the hash_block_tokens was called with the correct arguments
            calls = [pytest.call(mock_hash_function, None, block, None) for block in expected_blocks]
            mock_hash_block_tokens.assert_has_calls(calls, any_order=False)

            # Verify the result
            assert result == [f"hash_{block}" for block in expected_blocks]

def test_hash_request_tokens_no_prefix_incomplete_block(mock_hash_function, mock_request):
    """Test handling of incomplete blocks."""
    block_size = 4
    expected_blocks = [[1, 2, 3, 4], [5, 6, 7, 8]]

    with patch("vllm.v1.core.kv_cache_utils.need_extra_keys", return_value=False):
        with patch("vllm.v1.core.kv_cache_utils.hash_block_tokens") as mock_hash_block_tokens:
            mock_hash_block_tokens.side_effect = lambda hf, pb, tokens, ek: f"hash_{tokens}"

            result = hash_request_tokens_no_prefix(mock_hash_function, block_size, mock_request)

            # Verify the hash_block_tokens was called with the correct arguments
            calls = [pytest.call(mock_hash_function, None, block, None) for block in expected_blocks]
            mock_hash_block_tokens.assert_has_calls(calls, any_order=False)

            # Verify the result
            assert result == [f"hash_{block}" for block in expected_blocks]

def test_hash_request_tokens_no_prefix_with_extra_keys(mock_hash_function, mock_request):
    """Test functionality when extra keys are required."""
    block_size = 3
    expected_blocks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    with patch("vllm.v1.core.kv_cache_utils.need_extra_keys", return_value=True):
        with patch("vllm.v1.core.kv_cache_utils.generate_block_hash_extra_keys") as mock_generate_extra_keys:
            with patch("vllm.v1.core.kv_cache_utils.hash_block_tokens") as mock_hash_block_tokens:
                mock_generate_extra_keys.side_effect = lambda req, start, end, idx: (f"extra_keys_{start}_{end}", idx + 1)
                mock_hash_block_tokens.side_effect = lambda hf, pb, tokens, ek: f"hash_{tokens}_with_{ek}"

                result = hash_request_tokens_no_prefix(mock_hash_function, block_size, mock_request)

                # Verify the generate_block_hash_extra_keys was called
                extra_key_calls = [
                    pytest.call(mock_request, start, start + block_size, idx)
                    for idx, start in enumerate(range(0, len(mock_request.all_token_ids), block_size))
                    if start + block_size <= len(mock_request.all_token_ids)
                ]
                mock_generate_extra_keys.assert_has_calls(extra_key_calls, any_order=False)

                # Verify the hash_block_tokens was called with the correct arguments
                calls = [
                    pytest.call(mock_hash_function, None, block, f"extra_keys_{start}_{start + block_size}")
                    for start, block in zip(range(0, len(mock_request.all_token_ids), block_size), expected_blocks)
                ]
                mock_hash_block_tokens.assert_has_calls(calls, any_order=False)

                # Verify the result
                assert result == [f"hash_{block}_with_extra_keys_{start}_{start + block_size}" for start, block in zip(range(0, len(mock_request.all_token_ids), block_size), expected_blocks)]