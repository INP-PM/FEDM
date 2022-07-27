#!/bin/bash

# Run unit tests
python3 -m pytest tests/unit_tests -v

# Run integrated tests
# (run individually as FEniCS must reset between runs)
python3 -m pytest tests/integrated_tests/time_of_flight
python3 -m pytest tests/integrated_tests/streamer_discharge
