# main.py

from logger import Logger, log, quiet

# Using the Logger class as a context manager
with NoLogger():
    log("This will not be printed due to quiet being True")

# Using the global variable quiet
print(quiet)  # Should print False, as it is restored after the context manager block

# Using the log function without the context manager
log("This will be printed because quiet is False")

