from omegaconf import OmegaConf

config_file = "/home/dyvm6xra/dyvm6xrauser08/gh/vllm_project/vllm/vllm_omni/model_executor/stage_configs/qwen2_5_omni.yaml"
config_data = OmegaConf.load(config_file)

stage_configs = config_data.stage_args

for stage_config in stage_configs:
    print(stage_config.stage_id)