"""Module for representing documents in Dynamic RAG."""

from drag.logging import logger
from drag.utils import get_block_allocator, get_sampling_param
from vllm import LLM


class Document:
    _count = -1

    def __init__(self, text: str, llm: LLM):
        self.doc_id = Document.next()

        self.text = text
        self.llm = llm  # handle prefilling

        self.token_str = None
        self.token_ids = None
        # self.slot_mapping = None
        self.num_blocks = None
        self.block_table = None

        self.encode()
        self.prefill()

    def encode(self):
        """
        Encode the document text to token ids.
        :return:
        """
        tokenizer_grp = self.llm.llm_engine.tokenizer
        self.token_ids = tokenizer_grp.encode(self.text)
        logger.debug(f"Token ids - Length {len(self.token_ids)}\n" f"Raw : {self.token_ids}")
        self.token_str = tokenizer_grp.tokenizer.batch_decode(self.token_ids)  # skip_special_tokens=False
        logger.debug(f"Token str - Length {len(self.token_str)}\n" f"Raw : {self.token_str}")

    def prefill(self):
        """
        Prefill the document. This function will invoke the LLM to generate the token ids and slot mapping.
        Then the corresponding kv cache can be accessed via the slot ids.
        :return:
        """
        outputs = self.llm.generate([self.text], get_sampling_param("prefill"))
        logger.debug(f"prefill output - {outputs[0].outputs[0].text}")
        ctx = self.llm.llm_engine.scheduler_contexts[0]
        self.block_table = list(ctx.seq_group_metadata_list[0].block_tables.values())[0]
        self.num_blocks = len(self.block_table)

        # mark the blocks as special
        get_block_allocator(self.llm).evictor.add_special(self.block_table)

    @classmethod
    def next(cls):
        cls._count += 1
        return cls._count

    def destroy(self):
        """
        Destroy the document.
        :return:
        """
        if self.block_table is not None:
            # TODO: make sure no other document points the same block ids
            # remove the special block ids, so that they can be evicted
            get_block_allocator(self.llm).evictor.remove_special(self.block_table)

    def __str__(self):
        return f"doc id: {self.doc_id}\n" f"block table: {self.block_table}"
