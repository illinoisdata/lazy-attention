Use cuda/12.4.0 to build the project.

Note: delta cluster does not support `ncu`, since we do not have sudo access. Use `nsys` instead.

`ncu`: finer-grained profiling tool
`nsys`: system-wide profiling tool

## Basic commands

```shell
# run app with ncu, export the report to .report, check in nsight-compute
ncu --target-processes all --export output.report --metrics regex:sm__inst_executed_pipe_* --page raw --set full pytest ./your_app

# run the app with nsys, export the report to .qdrep, check in nsight-system
nsys profile -t cuda,nvtx ./your_app
```

## Specialize

For reference

```shell
ncu --target-processes all --export output.report --metrics regex:sm__inst_executed_pipe_* --page raw --set full python tests/test_overhead.py

nsys profile --trace=cuda,cudnn,cublas,nvtx,osrt -o output python tests/test_overhead.py
```