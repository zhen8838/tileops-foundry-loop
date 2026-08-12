"""Runtime twin of the exact production TileOps candidate."""

import authored_hir as sem
from tilefoundry.runtime import runtime_func, runtime_module

# Replace this placeholder with an import of the production TileOps kernel
# proposed by the PR.


@runtime_module(sem.OperatorModule)
class OperatorRuntime:
    @runtime_func
    def kernel(self, x):
        raise NotImplementedError("call the production TileLang path")
