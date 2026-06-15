"""
Centralized logging configuration
Uses loguru for structured logging
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


# Remove default handler
logger.remove()


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: Optional[str] = None,
):
    """
    Setup centralized logging with loguru

    Args:
        log_file: Path to log file (optional)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files
        retention: How long to keep old logs
        format_string: Custom format string (optional)
    """
    # Default format with colors and structure
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Console handler (stdout)
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler (if specified)
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_file,
                format=format_string,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True,
            )

            logger.info(f"Logging to file: {log_file}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot write to log file {log_file}: {e}")
            logger.info("Continuing with console logging only (Docker best practice)")


def get_logger(name: str):
    """
    Get a logger instance with specified name

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# No default setup — caller must invoke setup_logger() explicitly
