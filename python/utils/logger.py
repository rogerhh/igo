# Global variable
quiet = False

class NoLogger:
    def __init__(self):
        # Save the initial value of quiet when the Logger is created
        self.old_quiet_value = quiet

    def __enter__(self):
        # Inside the context manager, set quiet to True
        global quiet
        quiet = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # On exiting the context manager, restore the old value of quiet
        global quiet
        quiet = self.old_quiet_value

# Global function
def log(*args):
    # If quiet is True, print the arguments; otherwise, do nothing
    if not quiet:
        print(*args)

