from gym.envs.registration import register

register(
    id='fishing-v0',
    entry_point='gym_fishing.envs:FishingEnv',
)


register(
    id='fishing-v1',
    entry_point='gym_fishing.envs:FishingCtsEnv',
)

register(
    id='fishing-v2',
    entry_point='gym_fishing.envs:FishingTippingEnv',
)

register(
    id='fishing-v4',
    entry_point='gym_fishing.envs:FishingModelError',
)



