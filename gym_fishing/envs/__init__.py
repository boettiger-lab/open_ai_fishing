from gym.envs.registration import register

from gym_fishing.envs.base_twospecies_fishing_env import (
    BaseCompetingPairFishingEnv as ts_fishing_env,
)
from gym_fishing.envs.fishing_cts_env import FishingCtsEnv
from gym_fishing.envs.fishing_env import FishingEnv
from gym_fishing.envs.fishing_model_error import FishingModelError
from gym_fishing.envs.fishing_tipping_env import FishingTippingEnv
from gym_fishing.envs.growth_models import (
    Allen,
    BevertonHolt,
    May,
    ModelUncertainty,
    Myers,
    NonStationary,
    Ricker,
)
from gym_fishing.envs.trophic_triangle_env import trophicTriangleEnv

register(
    id="trophic-triangle-v0",
    entry_point="gym_fishing.envs:trophicTriangleEnv",
)

register(
    id="trophic-triangle-v1",
    entry_point="gym_fishing.envs:trophicTriangleRandEnv",
)

register(
    id="tsfishing-v0",
    entry_point="gym_fishing.envs:ts_fishing_env",
)

register(
    id="fishing-v0",
    entry_point="gym_fishing.envs:FishingEnv",
)

register(
    id="fishing-v1",
    entry_point="gym_fishing.envs:FishingCtsEnv",
)

register(
    id="fishing-v2",
    entry_point="gym_fishing.envs:FishingTippingEnv",
)

register(
    id="fishing-v4",
    entry_point="gym_fishing.envs:FishingModelError",
)


register(
    id="fishing-v5",
    entry_point="gym_fishing.envs:Allen",
)

register(
    id="fishing-v6",
    entry_point="gym_fishing.envs:BevertonHolt",
)

register(
    id="fishing-v7",
    entry_point="gym_fishing.envs:May",
)

register(
    id="fishing-v8",
    entry_point="gym_fishing.envs:Myers",
)

register(
    id="fishing-v9",
    entry_point="gym_fishing.envs:Ricker",
)

register(
    id="fishing-v10",
    entry_point="gym_fishing.envs:NonStationary",
)

register(
    id="fishing-v11",
    entry_point="gym_fishing.envs:ModelUncertainty",
)
