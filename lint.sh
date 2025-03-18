#!/bin/sh

set -e

echo "Formatting..."
echo "--- Ruff ---"
ruff format wavetrainer
echo "--- isort ---"
isort wavetrainer

echo "Checking..."
echo "--- Flake8 ---"
flake8 wavetrainer
echo "--- pylint ---"
pylint wavetrainer
echo "--- mypy ---"
mypy wavetrainer
echo "--- Ruff ---"
ruff check wavetrainer
echo "--- pyright ---"
pyright wavetrainer
