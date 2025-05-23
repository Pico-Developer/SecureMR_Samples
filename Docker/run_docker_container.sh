#!/usr/bin/env bash

# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -----------------------------------------------------------------------------
# Run the Docker Container
#
# Arguments:
# N/A
#
# Example Call:
#    bash run_docker_container.sh
# -----------------------------------------------------------------------------

adb kill-server
adb -a nodaemon server start &> /dev/null &
docker-compose up -d

docker exec -it securemr_tool bash
