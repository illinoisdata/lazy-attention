#!/bin/bash

mkdir ChatRAG-Bench/
huggingface-cli download nvidia/ChatRAG-Bench --repo-type dataset --local-dir ChatRAG-Bench/

mkdir Hotpot-Bench/
wget -O Hotpot-Bench/hotpot_test_v1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json