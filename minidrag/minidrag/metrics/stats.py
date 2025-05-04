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
    # For dynamic prefix cache, the number of blocks that were added to the
    doc_requests: int = 0
    doc_queries: int = 0
    doc_hits: int = 0