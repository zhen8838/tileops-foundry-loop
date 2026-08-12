"""Correct route metadata after the base replay leaves its env override set."""

import json
import os
from pathlib import Path

from tileops.kernels.gated_deltanet.gated_deltanet_prefill import (
    _prefill_auto_cp_local_chunks,
    _prefill_should_partition,
)


source = Path(".round3-cu132-artifacts/manifest-profile-cu132.jsonl")
target = Path(".round3-cu132-artifacts/manifest-profile-cu132-corrected.jsonl")
records = []
os.environ.pop("TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS", None)
os.environ.pop("TILEOPS_GDN_PREFILL_CP_MAX_LOCAL_CHUNKS", None)
for line in source.read_text().splitlines():
    record = json.loads(line)
    shape = record["shape"]
    num_chunks = shape["seq_len"] // shape["chunk_size"]
    local_chunks = _prefill_auto_cp_local_chunks(num_chunks, shape["heads"])
    record["route"]["candidate_max_local_chunks"] = local_chunks
    record["route"]["candidate_partition"] = _prefill_should_partition(
        shape["seq_len"], num_chunks, shape["heads"], local_chunks, False
    )
    records.append(record)
target.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
