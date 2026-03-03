import time
from collections import deque
from statistics import mean, median

from docling_ibm_models.tableformer.utils.mem_monitor import MemMonitor


class SingletonClass(type):
    r"""
    Generic singleton metaclass
    """

    def __init__(self, name, bases, dic):
        self._instance = None
        super().__init__(name, bases, dic)

    def __call__(cls, *args, **kwargs):
        # Create a singleton if needed
        if cls._instance is None:
            singleton = cls.__new__(cls)
            singleton.__init__(*args, **kwargs)
            cls._instance = singleton
        return cls._instance


class Profiler:
    r"""
    Application specific profiler
    Decompose the application into "sections". Each section is a label.
    The total time a section consumes is split into "intervals"
    Use the `begin`, `end` methods to mark the begining and end of an interval for
    a certain section
    """

    def __init__(self):
        self._section_dts = {}  # section name -> sum(section intervals)
        self._section_calls = {}  # section name -> number of invocations
        self._section_kB = {}  # section name -> max kB of used heap (resident set size)

        # section name -> beginning of the last interval
        self._last_begin = {}

        self._mem_monitor = MemMonitor()

    def begin(self, section_name, enable=True):
        r"""
        Mark the beginning of an interval

        Parameters
        ----------
        section_name : string
            Name of the section
        enable : bool
            The actual interval entry takes place only if enable is true

        Return
        ------
            True if the interval has actuall begun
        """
        if not enable:
            return False
        self._last_begin[section_name] = time.time()
        return True

    def end(self, section_name, enable=True):
        r"""
        Mark the end of an interval for a certain section

        Parameters
        ----------
        section_name : string
            Name of the section
        enable : bool
            The actual interval entry takes place only if enable is true

        Return
        ------
            True if the section name is valid and an interval for this section has already begun
            False otherwise
        """
        if not enable:
            return False
        if section_name not in self._last_begin:
            return False

        # Get memory
        kB = self._mem_monitor.get_memory()
        if isinstance(kB, dict):
            kB = kB["resident"]

        dt = time.time() - self._last_begin[section_name]
        if section_name not in self._section_dts:
            self._section_dts[section_name] = dt
            self._section_calls[section_name] = 1
            self._section_kB[section_name] = kB
        else:
            self._section_dts[section_name] += dt
            self._section_calls[section_name] += 1
            self._section_kB[section_name] = max(kB, self._section_kB[section_name])

        return True

    def get_data(self, section_names=None):
        r"""
        Return a dict with profiling data for the specified sections.

        Parameter
        ---------
        section_names : list of string
            List with the section names to get their accumulative dt
            If it is None, all sections are returned

        Return
        ------
        dict of dicts
            Outer key: section name
            Inner keys: "dt": Accumulative time for that section, "cells": Number of calls
        """
        # Filter the section names to apply
        filtered_names = list(
            filter(lambda x: x in section_names, self._section_dts.keys())
            if section_names is not None
            else self._section_dts.keys()
        )
        data = {}
        for section_name in filtered_names:
            data[section_name] = {
                "dt": self._section_dts[section_name],
                "calls": self._section_calls[section_name],
                "kB": self._section_kB[section_name],
            }
        return data


class AppProfiler(Profiler, metaclass=SingletonClass):
    r"""
    AppProfiler is a singleton of the Profiler for application wide usage
    """

    def __init__(self):
        super(AppProfiler, self).__init__()


class AggProfiler(metaclass=SingletonClass):
    r"""
    Generic wrapper of Profiler that enables aggregation of profiling statistics around Cycles

    - When a new cycle begins a new Profiler is created to keep the profiling data per section
    - Keep the last n cycles in a sliding window manner
    - At every time we can get profiling data about the last cycle and statistics over the last n
      cycles
    """

    def __init__(self, window_size=20):
        self._window_size = window_size
        # deque with up to the last "window_size" Profilers. The newest at index 0
        self._cycles = deque()

    def start_agg(self, enable=True):
        r"""
        Returns
        -------
        0: not enabled
        1: a new scope has started
        """
        if not enable:
            return 0

        # Add a new profiler
        self._cycles.appendleft(Profiler())
        # In case the deque has grown too much, remove the oldest Profiler
        if len(self._cycles) > self._window_size:
            self._cycles.pop()
        return 1

    def begin(self, section_name, enable=True):
        if not enable:
            return False
        if len(self._cycles) == 0:
            print("AggProfiler begin | Start Aggregator not initialized.")
            return False
        profiler = self._cycles[0]
        return profiler.begin(section_name)

    def end(self, section_name, enable=True):
        if not enable:
            return False
        if len(self._cycles) == 0:
            print("AggProfiler end | Start Aggregator not initialized.")
            return False
        profiler = self._cycles[0]
        return profiler.end(section_name)

    def get_data(self):
        r"""
        Get profiling data for:
        - The last cycle
        - Aggragated statistics (avg, median) per section and per metric across all cycles
        - The dt numbers for the mean/median is the average time for each section ACROSS the cycle
        - There is NO need to compute average by yourself.

        Returns
        -------
        dict with the structure:
        - window: int with the size of the time sliding window
        - last: dict with the metrics for the last cycle (as provided by the Profiler)
        - mean: dict with the mean metrics per section across the cycle
            - section_name
                - metric_name: mean of the metric values
        - median: dict with the median metrics per section across the cycle
            - section_name
                - metric_name: median of the metric values
        """
        last_data = self._cycles[0].get_data()
        data = {
            "window": len(self._cycles),
            "last": last_data,
            "mean": {},
            "median": {},
        }

        # Section -> metric -> [values]
        section_metric_values = {}

        # Collect the metrics
        for i, p in enumerate(self._cycles):
            p_data = p.get_data()
            for section_name, m_dict in p_data.items():
                for m_name, m_val in m_dict.items():
                    if section_name not in section_metric_values:
                        section_metric_values[section_name] = {}
                    s_metrics = section_metric_values[section_name]
                    if m_name not in s_metrics:
                        s_metrics[m_name] = []
                    s_metrics[m_name].append(m_val)

        # Aggregate the metrics
        for section_name, m_dict in section_metric_values.items():
            for m_name, m_values in m_dict.items():
                if section_name not in data["mean"]:
                    data["mean"][section_name] = {}
                if section_name not in data["median"]:
                    data["median"][section_name] = {}

                mean_v = mean(m_values)
                median_v = median(m_values)
                data["mean"][section_name][m_name] = mean_v
                data["median"][section_name][m_name] = median_v

        return data
