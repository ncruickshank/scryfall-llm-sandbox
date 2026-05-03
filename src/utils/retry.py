# written with assistance from ChatGPT
import time
def retry(fn, retries=3, delay=2, allowed_exceptions=(Exception,), context=""):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except allowed_exceptions as e:
            if attempt == retries:
                raise
            time.sleep(delay)