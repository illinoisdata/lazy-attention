"""
This backend will smartly route different task patterns to the appropriate Triton kernels.
1. Reorder search task:
    KV are frequently reused across different queries, making it beneficial to use KV as outer loop.

x. Others:
    General workload patterns that do not fit into the above categories. -> Use FlashAttention style
"""

# TODO(haocheng): Left it for future implementation.