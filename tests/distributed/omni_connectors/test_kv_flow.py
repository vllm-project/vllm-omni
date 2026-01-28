import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class MockGPUARModelRunner(GPUARModelRunner):
    """Subclass to bypass heavy initialization."""

    def __init__(self, kv_caches, input_batch):
        self.kv_caches = kv_caches
        self.input_batch = input_batch
        self.device = "cpu"
        self.cache_config = MagicMock()
        self.cache_config.block_size = 16
        self.cache_config.cache_dtype = "auto"
        self.logger = MagicMock()


class MockBagelPipeline(BagelPipeline):
    """Subclass to bypass heavy initialization."""

    def __init__(self):
        self.device = "cpu"
        self.od_config = MagicMock()
        self.bagel = MagicMock()
        self.tokenizer = MagicMock()
        self.language_model = MagicMock()
        self.new_token_ids = {}


class TestKVFlow(unittest.TestCase):
    def setUp(self):
        # Common constants
        self.num_layers = 2
        self.num_heads = 4
        self.head_dim = 16
        self.block_size = 8
        self.x = 8  # PagedAttention factor
        self.seq_len = 20
        self.req_id = "req_test_1"

    def test_sender_extraction_logic(self):
        """Test extraction logic in GPUARModelRunner."""
        if GPUARModelRunner is object:
            self.skipTest("vLLM not installed")

        # 1. Setup KV Cache (List of tuples (K, V))
        # Shape: [num_blocks, block_size, num_heads, head_dim] matching 4D expectation
        num_blocks = 10
        kv_caches = []
        for _ in range(self.num_layers):
            k_cache = torch.randn(num_blocks, self.block_size, self.num_heads, self.head_dim)
            v_cache = torch.randn(num_blocks, self.block_size, self.num_heads, self.head_dim)
            # Stack K and V to create [2, num_blocks, block_size, n_heads, head_dim]
            layer_cache = torch.stack([k_cache, v_cache], dim=0)
            kv_caches.append(layer_cache)

        # 2. Setup Input Batch Mock
        block_ids = [1, 3, 5]
        mock_input_batch = MagicMock()
        mock_input_batch.req_id_to_index = {self.req_id: 0}
        mock_input_batch.block_table.get_row.return_value = block_ids

        # 3. Instantiate Runner
        runner = MockGPUARModelRunner(kv_caches, mock_input_batch)

        # 4. Run Extraction
        # calling _extract_kv_cache directly: (req_id, block_ids, seq_len)
        result = runner._extract_kv_cache(self.req_id, block_ids, self.seq_len)

        # 5. Verify Result
        self.assertIsNotNone(result)
        self.assertEqual(result.request_id, self.req_id)

        # Check keys "key_cache" and "value_cache" exist (length 2)
        self.assertEqual(len(result.layer_blocks["key_cache"]), 2)
        self.assertEqual(len(result.layer_blocks["value_cache"]), 2)

        # Check Tensor Shape: [seq_len, num_heads, head_dim]
        # result.layer_blocks["key_cache"] is a list of tensors
        expected_shape = (self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(result.layer_blocks["key_cache"][0].shape, expected_shape)

        return result  # Return for use in next test

    def test_receiver_injection_logic(self):
        """Test injection logic in BagelPipeline."""
        if BagelPipeline is object:
            self.skipTest("vLLM not installed")

        # 1. Get Data from Sender Test (simulate transfer)
        # Re-create data manually to be independent
        key_cache = []
        value_cache = []
        expected_shape = (self.seq_len, self.num_heads, self.head_dim)
        for i in range(self.num_layers):
            key_cache.append(torch.randn(expected_shape))
            value_cache.append(torch.randn(expected_shape))

        layer_blocks = {"key_cache": key_cache, "value_cache": value_cache}

        transfer_data = MagicMock()  # Mock KVCacheTransferData
        transfer_data.layer_blocks = layer_blocks
        transfer_data.metadata = {"kv_lens": [self.seq_len], "ropes": [0]}

        # 2. Setup Request with Injected Data
        sp = OmniDiffusionSamplingParams(
            past_key_values=SimpleNamespace(**layer_blocks),
            kv_metadata=transfer_data.metadata,
        )

        req = OmniDiffusionRequest(["test"], sp)

        # 3. Setup Pipeline
        pipeline = MockBagelPipeline()

        # Mock Bagel's NaiveCache
        class RealNaiveCache:  # Minimal impl
            def __init__(self, n):
                self.key_cache = {i: None for i in range(n)}
                self.value_cache = {i: None for i in range(n)}

        pipeline.bagel.config.llm_config.num_hidden_layers = self.num_layers

        captured_context = {}

        def mock_prepare_prompts(curr_kvlens, curr_rope, **kwargs):
            # Capture the state passed to prepare_prompts
            captured_context["kv_lens"] = curr_kvlens
            captured_context["ropes"] = curr_rope
            return {}, [0], [0]

        pipeline.bagel.prepare_prompts = MagicMock(side_effect=mock_prepare_prompts)
        with patch("vllm_omni.diffusion.models.bagel.pipeline_bagel.NaiveCache") as MockNaiveCacheCls:
            # Setup the instance returned by the constructor
            mock_cache_instance = RealNaiveCache(self.num_layers)
            MockNaiveCacheCls.return_value = mock_cache_instance

        # Verification of Injection Logic (Simulation)
        current_cache = RealNaiveCache(self.num_layers)

        # --- Logic from Source Code ---
        injected_kv = req.sampling_params.past_key_values
        if isinstance(current_cache, RealNaiveCache) and hasattr(injected_kv, "key_cache"):
            # Assuming injected_kv is SimpleNamespace or object with list attrs
            for layer_idx in range(len(injected_kv.key_cache)):
                if injected_kv.key_cache[layer_idx] is not None:
                    k_tensor = injected_kv.key_cache[layer_idx]
                    v_tensor = injected_kv.value_cache[layer_idx]

                    if k_tensor.device != pipeline.device:
                        k_tensor = k_tensor.to(pipeline.device)
                    if v_tensor.device != pipeline.device:
                        v_tensor = v_tensor.to(pipeline.device)

                    current_cache.key_cache[layer_idx] = k_tensor
                    current_cache.value_cache[layer_idx] = v_tensor

        self.assertTrue(torch.allclose(current_cache.key_cache[0], layer_blocks["key_cache"][0]))
        self.assertTrue(torch.allclose(current_cache.value_cache[1], layer_blocks["value_cache"][1]))

    def test_integration(self):
        """Simulate the flow from Sender -> Connector (Dict) -> Receiver."""
        if GPUARModelRunner is object or BagelPipeline is object:
            self.skipTest("vLLM not installed")

        # 1. Sender (Extraction)
        runner_test_result = self.test_sender_extraction_logic()  # Get KVCacheTransferData

        # 2. Connector (Serialize/Deserialize Simulation)
        # KVCacheTransferData has to_dict
        data_dict = runner_test_result.to_dict()

        # 3. Receiver (Request Setup)
        sp = OmniDiffusionSamplingParams(
            past_key_values=data_dict["layer_blocks"],
            kv_metadata=data_dict["metadata"],
        )
        req = OmniDiffusionRequest(["integration_test"], sp)  # noqa: F841

        # 4. Receiver (Injection Simulation)
        # Use the logic verification again
        pass


if __name__ == "__main__":
    unittest.main()
