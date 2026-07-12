# Vendored from chatterbox.models.utils.AttrDict (4 lines inlined).
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


CFM_PARAMS = AttrDict(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
        "reg_loss_type": "l1",
    }
)
