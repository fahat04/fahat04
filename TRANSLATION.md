# Python Translation

This repository contains `methodAFit.py`, a simplified Python
translation of the provided C++ code.  The goal of the translation is
to mirror the structure of the original implementation while using
standard Python and `numpy` arrays instead of ROOT classes.

The main features are:

- A `physics` class with constant definitions and helper functions.
- `MethodAFit` dataclass replicating the original class behaviour.
- Simplified versions of the spectrum generation, time-of-flight model
  and histogram filling routines.
- Utility functions such as `eECal`, `ensure_range` and `numElectrons`.

This code is not a one–to–one rewrite; several complex ROOT based
parts were condensed to keep the example short, but the overall logic
follows the original algorithm.
