"""`LLM.generate` must forward the documents, whichever prompt form is used.

`prompts=` and the legacy `prompt_token_ids=` take different branches inside
`generate`, and `document_seqs` is orthogonal to both -- but it used to be bound
inside one of them, so the legacy path first raised `NameError` and then, once
that was papered over with a default, silently generated without any documents.

`generate` is driven here against a stand-in `self`: everything it touches is
recorded rather than executed, so the test needs no engine and no GPU.
"""
from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams

from lazy.entrypoints.llm import LazyLLM

DOCUMENTS = [["doc a", "doc b"]]


class RecordingLLM(SimpleNamespace):
    """Just enough of `LLM` for `generate` to run to the call we care about."""

    def __init__(self):
        super().__init__()
        self.recorded = {}
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(
            runner_type="generate", supported_runner_types=["generate"]))
        self.engine_class = SimpleNamespace(
            validate_outputs=lambda outputs, _kind: outputs)

    def _convert_v1_inputs(self, prompts, prompt_token_ids):
        return [{"prompt_token_ids": ids} for ids in prompt_token_ids]

    def _validate_and_add_requests(self, **kwargs):
        self.recorded = kwargs

    def _run_engine(self, use_tqdm):
        return []


@pytest.mark.unit
def test_documents_reach_the_engine_for_text_prompts():
    llm = RecordingLLM()

    LazyLLM.generate(llm,
                     prompts=["question?"],
                     sampling_params=SamplingParams(max_tokens=1),
                     document_seqs=DOCUMENTS)

    assert llm.recorded["document_seqs"] == DOCUMENTS


@pytest.mark.unit
def test_documents_reach_the_engine_for_legacy_token_ids():
    """The deprecated prompt_token_ids path is still a supported API."""
    llm = RecordingLLM()

    LazyLLM.generate(llm,
                     prompt_token_ids=[[1, 2, 3]],
                     sampling_params=SamplingParams(max_tokens=1),
                     document_seqs=DOCUMENTS)

    assert llm.recorded["document_seqs"] == DOCUMENTS


@pytest.mark.unit
def test_no_documents_stays_none():
    llm = RecordingLLM()

    LazyLLM.generate(llm,
                     prompts=["question?"],
                     sampling_params=SamplingParams(max_tokens=1))

    assert llm.recorded["document_seqs"] is None
