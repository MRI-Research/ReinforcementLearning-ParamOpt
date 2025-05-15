import logging
from pathlib import Path

def setup_logger(log_file: str) -> logging.Logger:
    """
    Configure and return a logger that writes to the specified file.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(log_file)        
    logger.setLevel(logging.DEBUG)               

    if not logger.handlers:   
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(levelname)-8s %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    return logger
