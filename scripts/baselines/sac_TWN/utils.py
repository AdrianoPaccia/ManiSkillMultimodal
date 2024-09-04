import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save the model checkpoints"""
    modes: tuple = ('rgb', 'depth')
    """to change the modes used"""

    # Env specific arguments
    env_id: str = "PushCube-v1"
    """the environment id of the task"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    num_eval_steps: int = 50
    """the number of steps to take in evaluation environments"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""
    eval_freq: int = 100_000
    """evaluation frequency in terms of environment steps"""
    save_train_video_freq: Optional[int] = 1
    """frequency to save training videos in terms of environment steps"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cpu"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""

    # multimodal loss coefficients
    MM_coeff: float = 1.0
    TF_coeff: float = 1.0
    TC_coeff: float = 1.0
    NTC_coeff: float = 1.0
    """cefficients for the representation loss"""




@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.obs = torch.zeros((buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((buffer_size, num_envs) + env.single_action_space.shape).to(storage_device)
        self.logprobs = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.rewards = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.dones = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.values = torch.zeros((buffer_size, num_envs)).to(storage_device)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )



'''
obs: torch.Tensor
img: torch.Tensor
next_obs: torch.Tensor
next_img: torch.Tensor
'''

########## MULTIMODAL
@dataclass
class ReplayBufferMultimodalSample:
    obs: tuple[torch.Tensor]
    next_obs: tuple[torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBufferMultimodal:
    def __init__(self, obs_shape:tuple, act_shape:tuple, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device, frames:int = 2):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device

        obs_shape = [(buffer_size, num_envs) + (frames, *s) for s in obs_shape]

        self.obs = [torch.zeros(s).to(storage_device) for s in obs_shape]
        self.next_obs = [torch.zeros(s).to(storage_device) for s in obs_shape]
        self.actions = torch.zeros((buffer_size, num_envs) + act_shape).to(storage_device)
        self.logprobs = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.rewards = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.dones = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.values = torch.zeros((buffer_size, num_envs)).to(storage_device)

    def add(self, obs: tuple[torch.Tensor], next_obs: tuple[torch.Tensor],
            action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):

        for i, (o, on) in enumerate(zip(obs,next_obs)):
            self.obs[i][self.pos] = o.to(self.storage_device)
            self.next_obs[i][self.pos] = on.to(self.storage_device)

        self.actions[self.pos] = action.to(self.storage_device)
        self.rewards[self.pos] = reward.to(self.storage_device)
        self.dones[self.pos] = done.to(self.storage_device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferMultimodalSample(
            obs=tuple([o[batch_inds, env_inds].to(self.sample_device) for o in self.obs]),
            next_obs=tuple([o[batch_inds, env_inds].to(self.sample_device) for o in self.next_obs]),
            #obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            #img=self.img[batch_inds, env_inds].to(self.sample_device),
            #next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            #next_img=self.next_img[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )

def get_distance(x_1, x_2, type='euclidean'):
    if type == 'euclidean':
        return torch.sum((x_1 - x_2) ** 2, dim=1)
    elif type == 'cosine':
        return F.cosine_similarity(x_1, x_2, dim=1)
    elif type == 'L1':
        return torch.sum(torch.abs(x_1 - x_2), dim=1)
    else:
        ValueError('The type of distance must be either "euclidean" or "cosine"')


