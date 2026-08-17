# Sourced: the `--env` arguments that put a container's compiled-kernel caches on
# the shared `/ci-cache` mount. Without them TileLang caches into the container's
# own `/root/.tilelang/cache`, so every round recompiles what the last one built
# and loses it on recreate. The directory is namespaced by TileLang build and
# entries commit by atomic rename, so several containers can share it. Passed on
# every `docker exec` as well as at creation, so an older container needs no
# recreate.

container_cache_env=(
    --env TILELANG_CACHE_DIR=/ci-cache/tilelang
    --env TILELANG_TMP_DIR=/ci-cache/tilelang/tmp
    --env TRITON_CACHE_DIR=/ci-cache/triton
)
