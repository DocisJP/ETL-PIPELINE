import time
import psutil
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        logger.info(f"Function '{func.__name__}' execution time: {execution_time:.2f} seconds")
        logger.info(f"Function '{func.__name__}' memory usage: {memory_used:.2f} MB")

        return result
    return wrapper

def log_overall_performance(start_time, start_memory):
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

    total_execution_time = end_time - start_time
    total_memory_used = end_memory - start_memory

    logger.info(f"Total ETL process execution time: {total_execution_time:.2f} seconds")
    logger.info(f"Total ETL process memory usage: {total_memory_used:.2f} MB")

def get_current_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB