#!/bin/bash
set -x
bash ./setup_models.sh
bash ./docker/build-gpu.sh