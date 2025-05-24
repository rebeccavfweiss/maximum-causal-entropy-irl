import logging
import os

def create_logger(filename:str="experiment_log") -> None:
    """
    Utility function to create a logger that logs into a given file
    
    Parameters
    ----------
    filename : str
        log file name
    """
    
    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)
    # Configure logging
    logging.basicConfig(
        filename=f"logs/{filename}.txt",
        filemode="w",  # Overwrite on each run. Use "a" to append.
        format="%(asctime)s - %(message)s",
        level=logging.INFO
    )

def log(msg:str, verbose:bool=False) -> None:
    """
    Logging function that can also print to the console if desired

    Parameters
    ----------
    msg : str
        message to log
    verbose : bool
        whether message should be printed to the console
    """
    logging.info(msg)
    if verbose:
        print(msg)