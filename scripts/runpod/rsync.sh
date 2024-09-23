#!/bin/bash

set -e

IP=66.114.112.70

folders=( "llm" "runpod" "pyproject.toml" )

for folder in "${folders[@]}"; do
  rsync -rvz -e 'ssh -p 59016 -i ~/.ssh/runpod' --progress --exclude '*.pt' --exclude '__pycache__*' ../$folder root@$IP:/workspace
done
