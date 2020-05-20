#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Run this script at project root by ".linter.sh" before you commit.
echo "Running isort..."
isort -y -sp .

echo "Running black..."
black -l 88 .

echo "Running flake..."
flake8 . --max-complexity=12 --max-line-length=88 --select=C,E,F,W,B,B950,BLK --ignore=E203,E231,E501,W503 

command -v arc > /dev/null && {
  echo "Running arc lint ..."
  arc lint
}
