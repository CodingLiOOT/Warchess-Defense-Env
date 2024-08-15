from loguru import logger
from torch.utils.tensorboard import SummaryWriter

# Initialize global variables
_writer = None


def init_logger(log_dir):
    """Initialize loguru logger with specific file paths."""
    logger.add(
        f"{log_dir}/train.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level="INFO",
        filter=lambda record: record["extra"].get("name") == "train",
    )
    logger.add(
        f"{log_dir}/runtime.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level="INFO",
        filter=lambda record: record["extra"].get("name") == "runtime",
    )


def get_logger(name):
    """Get a logger with a specific bind name."""
    return logger.bind(name=name)


def init_writer(log_dir):
    """Initialize TensorBoard SummaryWriter."""
    global _writer
    if _writer is None:
        _writer = SummaryWriter(log_dir=log_dir)


def log_scalar(tag, value, step):
    """Log a scalar value to TensorBoard."""
    if _writer is not None:
        _writer.add_scalar(tag, value, step)


def close_writer():
    """Close the TensorBoard writer."""
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None
