#!/bin/bash
set -x
bash ./setup_models.sh
cd ./docker
bash ./build-gpu.sh