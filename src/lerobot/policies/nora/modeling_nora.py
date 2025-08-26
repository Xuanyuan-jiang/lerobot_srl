import torch
from torch import nn
from lerobot.policies.pretrained import PreTrainedPolicy
from nora import NoraModel   # 这里用你在 nora repo 里的加载方法

class NoraPolicy(PreTrainedPolicy, nn.Module):
    name = "nora"

    def __init__(self, config, pretrained_name_or_path=None, dataset_stats=None):
        super().__init__()
        self.config = config
        self.device = config.device

        # 加载 checkpoint
        if pretrained_name_or_path is not None:
            self.nora = NoraModel.load_from_checkpoint(pretrained_name_or_path).to(self.device).eval()
        else:
            raise ValueError("NORA policy must be loaded from a pretrained checkpoint.")

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, config, dataset_stats=None):
        return cls(config, pretrained_name_or_path, dataset_stats)

    def reset(self):
        # 如果有 hidden state 可以在这里 reset
        pass

    def forward(self, obs, task=None, robot_type=None):
        """
        obs: 来自 record_loop -> predict_action
        """
        image = obs.get("observation.image")      # 确认下 dataset 的 obs key
        instruction = obs.get("observation.instruction", "")

        with torch.no_grad():
            actions = self.nora.inference(
                image=image,
                instruction=instruction,
                unnorm_key=None,
                unnormalizer=None,  # 可接 LeRobot 的 unnormalizer
            )
        return actions
