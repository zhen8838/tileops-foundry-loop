# Sourced, never executed: sets `container_cache_env` to the `--env` arguments
# that put a container's compiled-kernel caches on the shared `/ci-cache` mount.
#
# Without these, TileLang caches into the container's own `/root/.tilelang/cache`,
# so every round recompiles what every earlier round already built and loses all
# of it when its container is recreated. The mount is shared, the cache directory
# is namespaced by TileLang build, and entries are committed by atomic directory
# rename, which is what makes one cache safe for several containers -- the same
# arrangement TileOPs' own nightly runs on.
#
# `tileops-run` and the container script both pass these on every `docker exec`
# as well as at creation, so a container that predates this file still writes to
# the shared cache without being recreated.

container_cache_env=(
    --env TILELANG_CACHE_DIR=/ci-cache/tilelang
    --env TILELANG_TMP_DIR=/ci-cache/tilelang/tmp
    --env TRITON_CACHE_DIR=/ci-cache/triton
)
