# Block Attention vLLM

This version of Block Attention is integrated with vLLM backend and it shared the frontend from our Lazy Attention.

The main difference is 

1. We need to maintain an position array for each document
2. If documents are found in cache, position not matched and its `ref cnt == 0`, we can directly rotate,
   otherwise, we need to allocate new copies and copy these blocks, then rotate them


## Uasge

Similarly,

```python
vllm.generate(
prompts=[
    "Question 1",
    "Question 2",
],
document_seqs=[
    ["Doc1_1", "Doc1_2"],
    ["Doc2_1", "Doc2_2"],
]
)
```
