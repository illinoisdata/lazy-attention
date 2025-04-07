# minidrag/tests/conftest.py
import os
import sys

# add root directory to python path
vllm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, vllm_root)
minidrag_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../minidrag"))
sys.path.insert(0, minidrag_root)
