import torch

from vllm_omni.distributed.eplb.static_policy import compute_optimal_layout_greedy


def test_mapping_validity():
    """
    Verify that the log2phy mapping is a valid permutation of [0, num_experts-1].
    Ensures no experts are lost or duplicated during the re-layout process.
    """
    num_xpus = 2
    num_experts = 8
    layer_load = torch.rand(num_experts)

    log2phy = compute_optimal_layout_greedy(layer_load, num_xpus)

    assert log2phy.shape[0] == num_experts
    print("The new expert layout shape is equal to the number of global experts.")
    assert set(log2phy.tolist()) == set(range(num_experts))
    print("No expert repetition and no discarding")
    print("✓ Case 1 PASSED: Result is a valid permutation.")


def test_greedy_load_distribution():
    """
    Verify that the greedy algorithm effectively balances random loads.
    Even with stochastic input, the two experts with the highest loads
    must be assigned to different XPUs to ensure load balancing.
    """
    num_xpus = 2
    num_experts = 8
    experts_per_xpu = num_experts // num_xpus

    layer_load = torch.randn(num_experts).abs() * 100

    # The heaviest two experts need to be placed on different devices.
    _, top2_indices = torch.topk(layer_load, k=2)
    heavy_idx_1 = top2_indices[0].item()
    heavy_idx_2 = top2_indices[1].item()

    log2phy = compute_optimal_layout_greedy(layer_load, num_xpus)

    # Mapping from logical ID to device ID
    xpu_id_1 = log2phy[heavy_idx_1].item() // experts_per_xpu
    xpu_id_2 = log2phy[heavy_idx_2].item() // experts_per_xpu

    assert xpu_id_1 != xpu_id_2, (
        f"Load balancing failed: Expert {heavy_idx_1} (load {layer_load[heavy_idx_1]:.2f}) "
        f"and Expert {heavy_idx_2} (load {layer_load[heavy_idx_2]:.2f}) "
        f"were both assigned to XPU {xpu_id_1}."
    )
    print("✓ Case 2 PASSED: The heaviest experts have been placed on different devices.")


def test_expert_count_constraint():
    """
    Ensure each XPU handles an equal number of experts (experts_per_xpu).
    This is critical for synchronous All-to-All communication in MoE.
    """
    num_xpus = 4
    num_experts = 16
    layer_load = torch.randn(num_experts).abs()
    experts_per_xpu = num_experts // num_xpus

    log2phy = compute_optimal_layout_greedy(layer_load, num_xpus)

    counts = [0] * num_xpus
    for phy_slot in log2phy.tolist():
        xpu_id = phy_slot // experts_per_xpu
        counts[xpu_id] += 1

    for c in counts:
        assert c == experts_per_xpu
    print("✓ Case 3 PASSED: After the greedy algorithm, the number of experts is correct.")


if __name__ == "__main__":
    print("Running Static EPLB policy Tests...")
    print("=" * 60)

    print("\n[Running Case 1: Verify whether the output is a valid permutation. ]")
    test_mapping_validity()
    print("\n[Running Case 1: Verify that the greedy algorithm effectively balances random loads. ]")
    test_greedy_load_distribution()
    print("\n[Running Case 1: Verify the number of local experts after assignment. ]")
    test_expert_count_constraint()
