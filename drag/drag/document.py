"""Module for representing documents in Dynamic RAG."""

from drag.logging import logger
from drag.utils import get_block_size, get_ctx, get_evictor, get_sampling_param, get_tokenizer
from vllm import LLM


class Document:
    _count = -1

    def __init__(self, text: str, llm: LLM):
        self.doc_id = Document.next()

        self.text = text
        self.llm = llm

        # prepare padding
        self.block_size = get_block_size(self.llm)
        self.tokenizer = get_tokenizer(self.llm)
        self.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        # store tokens after padding
        self.token_str = None
        self.token_ids = None
        self.slot_mapping = None  # not used
        self.num_blocks = None
        self.block_table = None

        self.encode_and_padding()  # padding to fit with block size
        self.prefilling()  # prefilling the cache block and keep them in cache

    def encode_and_padding(self):
        """
        Encode the document text to token ids. If non-full block, padding with
        `pad_token`.
        :return: None
        """
        self.token_ids = self.tokenizer.encode(self.text)
        assert len(self.token_ids) > 0

        if len(self.token_ids) % self.block_size != 0:
            # padding
            num_pad_tokens = self.block_size - len(self.token_ids) % self.block_size
            logger.debug(f"Doc id {self.doc_id} pad {num_pad_tokens} tokens\n" f"Pad token id {self.pad_token_id}")
            self.token_ids.extend([self.pad_token_id] * num_pad_tokens)
        assert len(self.token_ids) % self.block_size == 0

        self.token_str = self.tokenizer.batch_decode(self.token_ids)
        logger.debug(f"Token str - Length {len(self.token_str)}\n" f"Raw : {self.token_str}")

    def prefilling(self):
        """
        Prefill the document. This function will invoke the LLM to generate KV
        cache for prompts.
        :return:
        """
        # directly feed token_ids rather than str,
        tokens_prompt = {
            "prompt_token_ids": self.token_ids,
        }
        logger.info(f'length of tokens prompt is {len(tokens_prompt["prompt_token_ids"])}')
        outputs = self.llm.generate([tokens_prompt], get_sampling_param("prefill"))
        # since token_ids not always equal to encode(decode(token_ids))
        # outputs = self.llm.generate(["".join(self.token_str)], get_sampling_param("prefill"))

        logger.debug(f"Doc id {self.doc_id}\n" f"Prefilling output - {outputs[0].outputs[0].text}")
        ctx = get_ctx(self.llm)
        # logger.debug(f"Block tables {ctx.seq_group_metadata_list[0].block_tables}")
        self.block_table = list(ctx.seq_group_metadata_list[0].block_tables.values())[0]
        self.num_blocks = len(self.block_table)

        logger.debug(
            f"Block table {self.block_table}\n"
            f"Number of allocated blocks {self.num_blocks},\n"
            f"Number of tokens {len(self.token_ids)}"
        )
        assert self.num_blocks == len(self.token_ids) // self.block_size

        # mark the blocks as special
        get_evictor(self.llm).add_special(self.block_table)

    def destroy(self):
        """
        Destroy the document.
        :return:
        """
        if self.block_table is not None:
            # TODO(haocheng): make sure no other document points the same block
            # ids when remove the special block ids, so that they can be evicted
            get_evictor(self.llm).remove_special(self.block_table)

    def __str__(self):
        return f"doc id: {self.doc_id}\n" f"block table: {self.block_table}"

    @classmethod
    def next(cls):
        cls._count += 1
        return cls._count
