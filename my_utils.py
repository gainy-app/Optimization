# Utility functions
from datetime import datetime

# Batch function
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# Date from string
def DateFromStr(str, format='%Y-%m-%d'):
    return datetime.strptime(str, format).date()

# Date from string
def DateToStr(dt, format='%Y-%m-%d'):
    return datetime.strftime(dt, format)
