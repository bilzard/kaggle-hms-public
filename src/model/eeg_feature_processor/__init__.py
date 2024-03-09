from .dual_per_channel import EegDualPerChannelFeatureProcessor
from .dual_per_channel_simple import EegDualPerChannelSimpleFeatureProcessor
from .dual_per_channel_v2 import EegDualPerChannelFeatureProcessorV2
from .dual_simple import EegDualSimpleFeatureProcessor

__all__ = [
    "EegDualPerChannelFeatureProcessor",
    "EegDualPerChannelSimpleFeatureProcessor",
    "EegDualSimpleFeatureProcessor",
    "EegDualPerChannelFeatureProcessorV2",
]
