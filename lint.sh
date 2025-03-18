#!/bin/sh

set -e

echo "Formatting..."
echo "--- Ruff ---"
ruff format wavetrain
echo "--- isort ---"
isort wavetrain

echo "Checking..."
echo "--- Flake8 ---"
flake8 wavetrain
echo "--- pylint ---"
pylint wavetrain
echo "--- mypy ---"
mypy wavetrain
echo "--- Ruff ---"
ruff check wavetrain
echo "--- pyright ---"
pyright wavetrain
