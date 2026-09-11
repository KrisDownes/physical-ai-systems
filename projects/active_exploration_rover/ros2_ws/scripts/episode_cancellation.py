"""Serialize episode cancellation with external turn submission, independent of ROS."""
from contextlib import contextmanager
import fcntl
import signal
from pathlib import Path


@contextmanager
def cancellation_lock(path):
    if path is None:
        yield
        return
    # A signal handler must never recursively acquire our own file lock.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        with Path(str(path)+'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def latch_cancel(path):
    with cancellation_lock(path):
        path.touch(exist_ok=True)
