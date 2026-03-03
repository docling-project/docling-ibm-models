import os
import platform
import re
from typing import Dict, Union


class MemMonitor:
    r"""
    Memory monitor for Linux

    It supports 2 approaches for extracting memory information:
    - linux-native: It parse the `/proc` pseudo-files. It is available only for Linux
    - psutil: Use the `psutil` library

    ## Linux-Native approach

    The linux-native approach implements 2 methods to extract the memory fields:

    1. The `get_memory()` method:

    - It is very fast
    - It parses the `/proc/<pid>/statm` pseudo-file
    - It Contains the following fields:
        size       (1) total program size
                   (same as VmSize in /proc/[pid]/status)
        resident   (2) resident set size
                   (same as VmRSS in /proc/[pid]/status)
        shared     (3) number of resident shared pages (i.e., backed by a file)
                   (same as RssFile+RssShmem in /proc/[pid]/status)
        text       (4) text (code)
        lib        (5) library (unused since Linux 2.6; always 0)
        data       (6) data + stack
        dt         (7) dirty pages (unused since Linux 2.6; always 0)


    2. The `get_memory_full()` method:

    - It is slower to parse but contains more detailed information
    - It uses regex to parse the `/proc/<pid>/status` pseudo-file
    - It contains the following fields:
        VmPeak: Peak virtual memory size.
        VmSize: Virtual memory size.
        VmLck: Locked memory size (see mlock(2)).
        VmPin: Pinned memory size (since Linux 3.2). These are pages that can't be moved because
               something needs to directly access physical memory.
        VmHWM: Peak resident set size ("high water mark").
        VmRSS: Resident set size.  Note that the value here is the sum of RssAnon, RssFile, and
               RssShmem.
        RssAnon: Size of resident anonymous memory.  (since Linux 4.5).
        RssFile: Size of resident file mappings.  (since Linux 4.5).
        RssShmem: Size of resident shared memory (includes System V shared memory, mappings from
                  tmpfs(5), and shared anonymous mappings).  (since Linux 4.5).
        VmData, VmStk, VmExe: Size of data, stack, and text segments.
        VmLib: Shared library code size.
        VmPTE: Page table entries size (since Linux 2.6.10).
        VmPMD: Size of second-level page tables (added in Linux 4.0; removed in Linux 4.15).
        VmSwap: Swapped-out virtual memory size by anonymous private pages; shmem swap usage is
                not included (since Linux 2.6.34).


    ## The psutil library

    - Apparently the psutil library parses the `/proc/<pid>/statm`
    - The memory_info() function returns the fields: rss, vms, shared, text, lib, data, dirty


    ## Field mappings

    These are the fields returned by psutil memory_info() and their mapping in the /proc files:
    (I put ? when I am not 100% about the mapping)

    | psutil  | /proc/$$/status    | /proc/$$/statm |
    |---------|--------------------|----------------|
    | rss     | VmRSS              | resident       |
    | vms     | VmSize             | size           |
    | shared  | RssFile + RssShmem | shared         |
    | text    | VmExe ?            | text           |
    | lib     | RssShmem ?         | lib            |
    | data    | VmData + VmStk     | data           |
    | dirty   | VmSwap ?           | dt             |

    """

    def __init__(self, enable=True):
        self._enable = enable
        self._pid = os.getpid()

        # Create regex for each memory field of the /proc/status pseudo-file
        self._status_fields = [
            "VmPeak",
            "VmSize",
            "VmLck",
            "VmPin",
            "VmHWM",
            "VmRSS",
            "RssAnon",
            "RssFile",
            "RssShmem",
            "VmData",
            "VmStk",
            "VmExe",
            "VmLib",
            "VmPTE",
            "VmPMD",
            "VmSwap",
        ]
        self._status_regex = {}
        for mem_field in self._status_fields:
            regex_str = r"({}:)(\s+)(\d*)(.*)".format(mem_field)
            self._status_regex[mem_field] = re.compile(regex_str)

    def get_memory_full(self) -> Union[Dict, int]:
        r"""
        - Parse /proc/<pid>status to get all memory info.
        - The method returns a dict with the fields self._status_fields
        - This method is SLOW. Unless you need the full memory info, better to use `get_memory`

        The returned values are in kB
        """
        if not self._enable:
            return -2
        if platform.system() != "Linux":
            return -1
        pid_fn = "/proc/{}/status".format(self._pid)

        # Dict to collect all memory fields
        memory = {}
        with open(pid_fn, "r") as fn:
            for ll in fn:
                for mem_field in self._status_fields:
                    regex = self._status_regex[mem_field]
                    m = regex.match(ll)
                    if m is not None:
                        memory[mem_field] = int(m.group(3))
                if len(memory) == len(self._status_fields):
                    break

        return memory

    def get_memory(self) -> Union[Dict, int]:
        r"""
        - Parse /proc/<pid>statm to get the most important memory fields
        - This is a fast implementation.
        - The method returns a dict with the fields:
            "size", "resident", "shared", "text", "lib", "data", "dt"
        - Check the documentation at the top for a mapping across the various fields

        The returned values are in kB
        """
        if not self._enable:
            return -2
        if platform.system() != "Linux":
            return -1
        pid_fn = "/proc/{}/statm".format(self._pid)

        # Dict to collect all memory fields
        memory = {}
        with open(pid_fn, "r") as fn:
            ll = fn.read()
            # The values are in pages
            # Each page is 4096 bytes (4kB)
            data = [int(x) << 2 for x in ll.split(" ")]
            memory = {
                "size": data[0],
                "resident": data[1],
                "shared": data[2],
                "text": data[3],
                "lib": data[4],
                "data": data[5],
                "dt": data[6],
            }
        return memory
