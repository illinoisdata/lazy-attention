import loguru
import sys

logger = loguru.logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")