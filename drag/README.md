# Dynamic RAG

TODO
- [ ] Measure accuracy
- [ ] Measure memory usage (RAM, GPU memory, pressure etc.)
- [ ] Quick plot script
- [ ] Onboard `RAGBench`
    - [ ] RAGCache: MMLU (MCQ) and Natural Questions (Short Answers)
    - [ ] Raptor: QuALITY, QASPER
- [ ] Baselines
    - [ ] Raptor (https://github.com/parthsarthi03/raptor)
    - [ ] Superposition Prompting (https://github.com/apple/ml-superposition-prompting)
    - [ ] RAGCache

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
bash scripts/bench_exp1.sh parrot,llmrag,pcrag narrativeqa
```

### Running on Slurm

```bash
sbatch --mail-user=${USER}@illinois.edu --mail-type="BEGIN,END" scripts/job_exp1.slurm
```

## Acknowledgement

```
@article{liu2024chatqa,
  title={ChatQA: Surpassing GPT-4 on Conversational QA and RAG},
  author={Liu, Zihan and Ping, Wei and Roy, Rajarshi and Xu, Peng and Lee, Chankyu and Shoeybi, Mohammad and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2401.10225},
  year={2024}}
```
