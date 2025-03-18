#!/bin/sh

set -e

PYTEST_CURRENT_TEST=1 pytest -s --cov-report=term --cov=wavetrain tests
coverage html -d coverage_html
