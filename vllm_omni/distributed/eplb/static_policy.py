import torch
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_static_eplb_weights(pipeline, all_layers_log2phy):
    if not isinstance(all_layers_log2phy, dict) or not all(
        isinstance(k, int) and isinstance(v, torch.Tensor) for k, v in all_layers_log2phy.items()
    ):
        raise TypeError("Invalid layout format. Expected Dict[int, torch.Tensor].")

    ep_group = get_ep_group()
    ep_device_group = ep_group.device_group
    ep_rank = ep_device_group.rank()
    ep_size = ep_device_group.size()

    logger.info(f"[rank {ep_rank}] [Static EPLB] start replacing static eplb layout.")
    for layer_idx, decoder_layer in enumerate(pipeline.model.layers):
        if layer_idx not in all_layers_log2phy:
            continue

        logger.debug(f"[rank {ep_rank}] [Static EPLB] layer{layer_idx} start replacing static eplb layout.")

        log2phy = all_layers_log2phy[layer_idx].to(decoder_layer.mlp.experts.w13_weight.device)
        decoder_layer.mlp.expert_layout = log2phy
        phy2log = torch.argsort(log2phy)

        num_global_experts = len(log2phy)
        experts_per_xpu = num_global_experts // ep_size

        start_idx = ep_rank * experts_per_xpu
        end_idx = (ep_rank + 1) * experts_per_xpu
        needed_logical_ids = phy2log[start_idx:end_idx]

        experts_module = decoder_layer.mlp.experts
        expert_param_names = ["w13_weight", "w2_weight"]
        for param_name in expert_param_names:
            assert hasattr(experts_module, param_name)
            local_tensor = getattr(experts_module, param_name).data

            global_tensor = ep_group.all_gather(local_tensor, dim=0)
            new_local_tensor = global_tensor[needed_logical_ids]
            local_tensor.copy_(new_local_tensor)


def compute_optimal_layout_greedy(layer_load: torch.Tensor, num_xpus: int = 4) -> torch.Tensor:
    num_experts = len(layer_load)
    experts_per_xpu = num_experts // num_xpus

    expert_info = [(load.item(), i) for i, load in enumerate(layer_load)]
    expert_info.sort(key=lambda x: x[0], reverse=True)

    xpu_loads = [0] * num_xpus
    xpu_buckets = [[] for _ in range(num_xpus)]

    for load, logical_id in expert_info:
        best_xpu = -1
        min_load = float("inf")

        for i in range(num_xpus):
            if len(xpu_buckets[i]) < experts_per_xpu and xpu_loads[i] < min_load:
                min_load = xpu_loads[i]
                best_xpu = i

        xpu_buckets[best_xpu].append(logical_id)
        xpu_loads[best_xpu] += load

    for i in range(num_xpus):
        xpu_buckets[i].sort()

    phy2log = []
    for bucket in xpu_buckets:
        phy2log.extend(bucket)

    log2phy = [0] * num_experts
    for phy_slot, logical_id in enumerate(phy2log):
        log2phy[logical_id] = phy_slot

    return torch.tensor(log2phy, dtype=torch.long)
