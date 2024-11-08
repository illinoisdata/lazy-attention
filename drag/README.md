# Dynamic RAG

TODO
- [ ] Asynchronously stream output to measure TTFT
- [ ] Measure memory usage (RAM, GPU memory, pressure etc.)
- [ ] Quick plot script
- [ ] Onboard `RAGBench`

## Benchmarks

Setup environment.

```bash
make virualenv
source .venv/bin/activate
make install
```

Install `vllm` (e.g., `cd .. && pip install -e .`).

Download datasets.

```bash
bash scripts/download_dataset.sh
```

### ChatRAG-Bench

For example:

```bash
python benchmarks/chatragbench.py --eval_dataset doc2dial \
            --data_folder ChatRAG-Bench/data --output_folder results/chatragbench \
            --rag_type=parrot
```

## Acknowledgement

```
@article{liu2024chatqa,
  title={ChatQA: Surpassing GPT-4 on Conversational QA and RAG},
  author={Liu, Zihan and Ping, Wei and Roy, Rajarshi and Xu, Peng and Lee, Chankyu and Shoeybi, Mohammad and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2401.10225},
  year={2024}}
```
