from stable_baselines.common.policies import FeedForwardPolicy
from .sb_policy_registry import PolicyRegistry
# Custom MLP policy of three layers of size 128 each

@PolicyRegistry.register("AGENT_1")
class AGENT_1(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(AGENT_1, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

