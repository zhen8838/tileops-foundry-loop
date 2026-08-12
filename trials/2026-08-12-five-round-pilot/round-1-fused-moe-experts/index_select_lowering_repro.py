from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import index_select
from tilefoundry.ir.core.module import Module
from tilefoundry.passes.transforms import HirToTirPass


@func
def select_routes(
    weights: Tensor[(4, 3), "f32"],
    route_ids: Tensor[(2,), "i32"],
) -> Tensor[(2, 3), "f32"]:
    return index_select(weights, route_ids, dim=0)


module = Module(
    name="index_select_lowering_repro",
    functions=(select_routes,),
    entry=select_routes.name,
)
HirToTirPass().run(module)
