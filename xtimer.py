# timer.py

import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self._history = None
        self._lap_history = None
        self._previous_lap_point = None

    def start(self, verbose = True):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
        if verbose:
            print('Timer started')
        self._previous_lap_point = self._start_time
        self._history = []
        self._lap_history = []
        self._history.append(self._start_time)

    def stop(self, verbose = True):
        """Stop the timer, and report amd return the total time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._previous_lap_point
        total_time = time.perf_counter() - self._start_time
        self._history.append(total_time)
        self._start_time = None
        if verbose:
            print(f"Elapsed time: {elapsed_time:0.3f} seconds")
            print(f"Total time: {total_time:0.3f} seconds")
        self._lap_history.append(elapsed_time)
        return total_time
    
    def total_t(self):
        return time.perf_counter() - self._start_time

    def lap(self, verbose = True):
        """report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        total_time = time.perf_counter() - self._start_time
        elapsed_time = time.perf_counter() - self._previous_lap_point
        self._previous_lap_point = time.perf_counter()
        self._history.append(total_time)
        if verbose:
            print(f"Elapsed time: {elapsed_time:0.3f} seconds (totally {self.total_t():0.3f}s)")
        self._lap_history.append(elapsed_time)
        return elapsed_time

    def reset(self, verbose = True):
        """Start a new timer"""
        if verbose:
            if self._start_time is None:
                self._start_time = time.perf_counter()
                print('Timer started since no start yet')
            else:
                self._start_time = time.perf_counter()
                print('Timer reset')
        self._history = []
        self._lap_history = []
        self._history.append(self._start_time)

    def report_history(self):
        """report the history"""
        if self._history is None or len(self._history) is 0:
            print('No history')
        else:
            for j, k in enumerate(self._history):
                print(f"Time point {j}: {k}")

    def report_lap_history(self):
        """report the history"""
        if self._lap_history is None or len(self._lap_history) is 0:
            print('No lap history')
        else:
            for j, k in enumerate(self._lap_history,1):
                print(f"Lap {j}: {k:0.3f} seconds")
