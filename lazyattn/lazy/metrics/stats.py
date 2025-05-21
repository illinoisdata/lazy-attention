import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

@dataclass
class PrefixCacheStats:
    """Stores prefix cache hit statistics."""
    # Whether reset_prefix_cache was invoked.
    reset: bool = False
    # The number of requests in this update.
    requests: int = 0
    # The number of queries in these requests. Note that "queries" here
    # means the number of blocks that were queried from the cache.
    queries: int = 0
    # The number of hits in these requests.
    hits: int = 0
    
    # # TODO(haocheng): if needed we can have a fine-grained one
    # # We dicrectly use the hits to calculate the hit rate.
    # fetch_blocks: int = 0.0
    # hit_blocks: int = 0.0
    
    

@dataclass
class BlockUsageStats:
    """Stores block usage statistics."""
    num_used_blocks: int = 0