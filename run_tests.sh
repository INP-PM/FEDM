#!/bin/bash

# Run unit tests
python3 -m pytest -v tests/unit_tests

# Run integrated tests
# These must be run independently, as FEniCS must reset between runs.
# `python3 -m pytest -v tests` will succeed on the first integrated test, but
# fail on the second/third.
python3 -m pytest -v tests/integrated_tests/time_of_flight
python3 -m pytest -v tests/integrated_tests/streamer_discharge
python3 -m pytest -v tests/integrated_tests/glow_discharge
