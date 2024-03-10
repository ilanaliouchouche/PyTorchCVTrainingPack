import logging


def get_logger(name: str = "fake_detector"):
    """
    Get logger with the specified name.

    Args:
        name (str): Name of the logger.
    
    Returns:
        logging.Logger: Logger with the specified name.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
