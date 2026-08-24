#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
"$repo_dir/scripts/build_tilefoundry_wheel.sh"
"$repo_dir/scripts/build_agent_image.sh"
