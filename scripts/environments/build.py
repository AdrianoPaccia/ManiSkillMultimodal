import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import os
from scripts.environments.wrappers import EnvMultimodalWrapper
from scripts.environments.noise.noise import ImageNoise, DepthNoise, SegmentNoise, ConfNoise
import yaml


def build_training_env(
        id:str,
        num_envs:int=1,
        obs_mode:str="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        device:str="gpu",
        **kwargs
    ):
    """
    Args:
        id: name of the environment
        num_envs: number of environments
        obs_mode: what tipe of observations
        control_mode: what tipe of actions
        render_mode: ["human", "rgb_array", "None"]
        device: ["cpu", "gpu"]
        capture_video:bool
        **kwarg: [run_name, partial_reset:bool]
    Returns: training environment
    """
    env_kwargs = dict(obs_mode=obs_mode, control_mode=control_mode, render_mode=render_mode, sim_backend=device, render_backend=device)
    envs = gym.make(id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=not kwargs['partial_reset'], **env_kwargs)
    game = kwargs['game']
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/configuration.yaml', "r") as file:
        kwargs = {**kwargs, **yaml.safe_load(file)[str(game)]}
    envs = EnvMultimodalWrapper(envs, **kwargs)
    return envs


def build_eval_env(
        id:str,
        num_envs:int=1,
        obs_mode:str="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        device:str="gpu",
        capture_video:bool=True,
        **kwargs
    ):
    """
    Args:
        id: name of the environment
        num_envs: number of environments
        obs_mode: what tipe of observations
        control_mode: what tipe of actions
        render_mode: ["human", "rgb_array", "None"]
        device: ["cpu", "gpu"]
        capture_video:bool
        **kwargs: [run_name, checkpoint_dir, num_steps, partial_reset:bool]

    Returns: testing environment

    """
    env_kwargs = dict(obs_mode=obs_mode, control_mode=control_mode, render_mode=render_mode, sim_backend=device, render_backend=device)
    envs = gym.make(id, num_envs=num_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    if capture_video:
        print(f"Saving eval videos to {kwargs['checkpoint_dir']}")
        envs = RecordEpisode(envs, output_dir=f"runs/{kwargs['run_name']}/train_videos", save_trajectory=True, trajectory_name="trajectory",
                                  max_steps_per_video=kwargs['eval_steps'], video_fps=30)

    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=not kwargs['partial_reset'], **env_kwargs)

    game = kwargs['game']
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/configuration.yaml', "r") as file:
        kwargs = {**kwargs, **yaml.safe_load(file)[str(game)]}

    noise_generators = build_noise_generators(
        game=kwargs['game'],
        types=['nonoise'],
        modes=list(envs.single_observation_space.spaces['sensor_data'][kwargs['sensor_data']['data_source']])
    )

    envs = EnvMultimodalWrapper(
        envs,
        noise_generators=noise_generators,
        **kwargs
    )
    return envs


def build_noise_generators(game, types, modes):
    res = {}
    for m in modes:
        if m=='rgb':
            ng = ImageNoise(game,types)
        elif m=='depth':
            ng = DepthNoise(game, types)
        elif m=='segmentation':
            ng = SegmentNoise(game, types)
        elif m=='state':
            ng = ConfNoise(game, types)
        else:
            raise ValueError(f'Mode {m} is not valid!')
        res[m] = ng
    return res
