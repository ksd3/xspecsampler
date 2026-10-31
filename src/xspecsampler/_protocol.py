"""Markers that frame the output of the worker subprocesses.

They are defined in a separate module so that importing the package does not
import the worker module.
"""

READY = ">XSPECSAMPLER_READY<"
RESULT_START, RESULT_END = ">XSS>", "<XSS<"
