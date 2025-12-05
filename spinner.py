import sys
import threading
import time


class Spinner:
    """
    Simple CLI spinner as a context manager.

    Usage:
        with Spinner("Doing stuff..."):
            do_expensive_thing()
    """

    def __init__(self, message="Working…", interval=0.1):
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _spin(self):
        frames = "|/-\\"
        i = 0
        sys.stdout.write(self.message + " ")
        sys.stdout.flush()
        while not self._stop.is_set():
            sys.stdout.write(frames[i % len(frames)])
            sys.stdout.flush()
            time.sleep(self.interval)
            sys.stdout.write("\b")
            i += 1
        # clear spinner frame
        sys.stdout.write(" \b")
        sys.stdout.flush()

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        # print newline after spinner so next prints look clean
        sys.stdout.write("\n")
        sys.stdout.flush()
