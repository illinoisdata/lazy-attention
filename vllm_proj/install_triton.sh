git clone https://github.com/triton-lang/triton.git
git checkout v3.2.0
cd triton

pip install ninja cmake wheel pybind11; # build-time dependencies
pip install -e python
