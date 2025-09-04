module load cuda/12.4.0
# PYTHONPATH=.:./promptcache bash scripts/bench_exp1.sh reullmrag 2wikimqa 
# --sample-requests 200 --max-concurrency 1
# PYTHONPATH=.:./promptcache bash scripts/bench_exp1.sh llmrag 2wikimqa 

# PYTHONPATH=.:./promptcache bash scripts/bench_exp1.sh blockattnrag 2wikimqa 
# PYTHONPATH=.:./promptcache bash scripts/bench_exp1.sh pcrag 2wikimqa
# PYTHONPATH=.:./promptcache bash scripts/bench_exp1.sh blockattnrag 2wikimqa 
PYTHONPATH=.:./promptcache bash scripts/bench_exp1.sh blockattnrag 2wikimqa 