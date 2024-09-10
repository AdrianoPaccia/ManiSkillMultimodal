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
    wandb_group: str = 'sac_CURL'
    """the entity (team) of wandb's project"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save the model checkpoints"""
    modes: str = 'rgb+depth'
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
    train_noise_types: str = 'nonoise'
    eval_noise_types: str = 'nonoise'
    train_noise_freq: float = 0.0
    eval_noise_freq: float = 0.0

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

########## MULTIMODAL
@dataclass
class ReplayBufferMultimodalSample:
    states: torch.Tensor
    next_states: torch.Tensor
    obs: tuple[torch.Tensor]
    next_obs: tuple[torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBufferMultimodal:
    def __init__(self, state_shape:tuple, obs_shape:tuple, act_shape:tuple, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device, frames:int = 2):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device

        obs_shape = [(buffer_size, num_envs) + (frames, *s) for s in obs_shape]

        self.states = torch.zeros((buffer_size, num_envs) + state_shape).to(storage_device)
        self.next_states = torch.zeros((buffer_size, num_envs) + state_shape).to(storage_device)
        self.obs = [torch.zeros(s).to(storage_device) for s in obs_shape]
        self.next_obs = [torch.zeros(s).to(storage_device) for s in obs_shape]
        self.actions = torch.zeros((buffer_size, num_envs) + act_shape).to(storage_device)
        self.logprobs = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.rewards = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.dones = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.values = torch.zeros((buffer_size, num_envs)).to(storage_device)

    def add(self, states, obs: tuple[torch.Tensor], next_states, next_obs: tuple[torch.Tensor],
            action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):

        self.states[self.pos] = states.to(self.storage_device)
        self.next_states[self.pos] = next_states.to(self.storage_device)


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
            states=self.states[batch_inds, env_inds].to(self.sample_device),
            next_states=self.next_states[batch_inds, env_inds].to(self.sample_device),
            obs=tuple([o[batch_inds, env_inds].to(self.sample_device) for o in self.obs]),
            next_obs=tuple([o[batch_inds, env_inds].to(self.sample_device) for o in self.next_obs]),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )

