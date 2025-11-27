#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (directory where this script lives)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

# Allow overriding Python via $PYTHON, default to python3
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python3 not found. Install Python 3.8+ and retry." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Reusing existing virtualenv at $VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip

if [ -f "$REPO_ROOT/requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  "$VENV_DIR/bin/pip" install -r "$REPO_ROOT/requirements.txt"
else
  echo "requirements.txt not found, installing core deps..."
  "$VENV_DIR/bin/pip" install pydicom numpy opencv-python tk
fi

echo
echo "Environment ready."
"$VENV_DIR/bin/python" -V

