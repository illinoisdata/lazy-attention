import pytest
from vllm.v1.core.kv_cache_utils import BlockHashType
from vllm.utils import sha256
from tests.v1.core.test_kv_cache_utils import make_request


class TestBlockHash:
    @pytest.mark.unit
    @pytest.mark.parametrize("hash_fn", [sha256, hash])
    def test_hash_request_tokens_no_prefix(self,hash_fn):
        # construct two requests with the two blocks
        request1 = make_request(
            request_id=0,
            prompt_token_ids=[_ for _ in range(6)],
        )

        request2 = make_request(
            request_id=0,
            prompt_token_ids=[_ for _ in range(3, 6)] + [_ for _ in range(3)],
        )

        block_size = 3
        from minidrag.core.kv_cache_utils import apply_patch, revert_patch
        apply_patch()
        from vllm.v1.core.kv_cache_utils import hash_request_tokens
        block_hashes_1 = hash_request_tokens(hash_fn, block_size, request1)
        assert len(block_hashes_1) == 2
        assert isinstance(block_hashes_1[0], BlockHashType)
        assert isinstance(block_hashes_1[1], BlockHashType)

        block_hashes_2 = hash_request_tokens(hash_fn, block_size, request2)
        assert len(block_hashes_2) == 2
        assert isinstance(block_hashes_2[0], BlockHashType)
        assert isinstance(block_hashes_2[1], BlockHashType)

        # assert the hash values are the same
        assert block_hashes_1[0].hash_value == block_hashes_2[1].hash_value
        assert block_hashes_1[1].hash_value == block_hashes_2[0].hash_value
        revert_patch()

        # Note(haocheng): after reverting the patch, the hash values should be different
        from vllm.v1.core.kv_cache_utils import hash_request_tokens
        block_hashes_1 = hash_request_tokens(hash_fn, block_size, request1)
        block_hashes_2 = hash_request_tokens(hash_fn, block_size, request2)
        assert block_hashes_1[0].hash_value != block_hashes_2[1].hash_value
        assert block_hashes_1[1].hash_value != block_hashes_2[0].hash_value
