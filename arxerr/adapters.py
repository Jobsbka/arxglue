"""
Adapters for external logging systems
"""

import logging
from typing import Optional, Any

def setup_std_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure integration with standard logging module"""
    logger = logging.getLogger('arxerr')
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def log_to_std(ctx: Any, logger: Optional[logging.Logger] = None):
    """Forward context logs to standard logging system"""
    logger = logger or setup_std_logging()
    
    if not hasattr(ctx, 'logs'):
        return
    
    for entry in ctx.logs:
        if entry.startswith("[ERROR]"):
            logger.error(entry[8:])
        elif entry.startswith("[WARNING]"):
            logger.warning(entry[10:])
        elif entry.startswith("[INFO]"):
            logger.info(entry[7:])
        elif entry.startswith("[DEBUG]"):
            logger.debug(entry[8:])
        else:
            logger.info(entry)