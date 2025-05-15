module load cuda/12.4.0
# PYTHONPATH=.:./promptcache LAZY_LOG=1 bash scripts/bench_exp1.sh llmrag,drag 2wikimqa # ,musique # ,llmrag,trragm1,pcrag
PYTHONPATH=.:./promptcache bash scripts/bench_exp1.sh llmrag,drag musique # , # ,llmrag,trragm1,pcrag
