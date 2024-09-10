import copy
import os
import random
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro
from models import ActorMultimodal as Actor
from models import SoftQNetworkMultimodal as SoftQNetwork

from utils import Args
from utils import ReplayBufferMultimodal as ReplayBuffer
from environments_multimodal.build import build_training_env, build_eval_env

from process import process_obs_dict
try:
    import matplotlib
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt


def train(**kwargs):
    '''
    Needs:
     + models: (actor, qf1_target, qf2_target)
     + ReplayBuffer
    Returns:

    '''
    envs.reset_mm(seed=args.seed)
    eval_envs.reset_mm(seed=args.seed)
    best_success_rate = 0.0
    global_step = 0
    global_update = 0
    learning_has_started = False
    alpha = kwargs['alpha']
    global_steps_per_iteration = args.num_envs * (args.steps_per_env)

    #image_reshape = envs.single_observation_space_mm[0].shape[1:]

    ## GLOBAL LOOP
    while global_step < args.total_timesteps:
        print(f"Global Step: {global_step}", end='\r')

        ## EVALUATION and SAVING
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            actor.eval()
            print("Evaluating")
            old_eval_obs, _ = eval_envs.reset_mm(seed=args.seed)
            eval_obs, _, _, _, _ = eval_envs.step_mm(eval_envs.action_space.sample())
            old_eval_obs = copy.deepcopy(eval_obs)
            eval_obs, _, _, _, _ = eval_envs.step_mm(eval_envs.action_space.sample())

            returns = []
            eps_lens = []
            successes = []
            failures = []

            #EVALUATION
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    obs_stack = process_obs_dict(eval_obs, old_eval_obs, args.modes, device)
                    action = actor.get_eval_action(obs_stack)
                    next_eval_obs, _, eval_terminations, eval_truncations, eval_infos = eval_envs.step_mm(action)
                    old_eval_obs = copy.deepcopy(eval_obs)
                    eval_obs = copy.deepcopy(next_eval_obs)

                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        eps_lens.append(eval_infos["final_info"]["elapsed_steps"][mask].cpu().numpy())
                        returns.append(eval_infos["final_info"]["episode"]["r"][mask].cpu().numpy())
                        if "success" in eval_infos:
                            successes.append(eval_infos["final_info"]["success"][mask].cpu().numpy())
                        if "fail" in eval_infos:
                            failures.append(eval_infos["final_info"]["fail"][mask].cpu().numpy())

            returns = np.concatenate(returns)
            eps_lens = np.concatenate(eps_lens)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {len(eps_lens)} episodes")
            if len(successes) > 0:
                successes = np.concatenate(successes)
                if writer is not None: writer.add_scalar("charts/eval_success_rate", successes.mean(), global_step)
                print(f"eval_success_rate={successes.mean()}")
            if len(failures) > 0:
                failures = np.concatenate(failures)
                if writer is not None: writer.add_scalar("charts/eval_fail_rate", failures.mean(), global_step)
                print(f"eval_fail_rate={failures.mean()}")

            print(f"eval_episodic_return={returns.mean()}")
            if writer is not None:
                writer.add_scalar("charts/eval_episodic_return", returns.mean(), global_step)
                writer.add_scalar("charts/eval_episodic_length", eps_lens.mean(), global_step)
            actor.train()
            if args.evaluate:
                break

            if args.save_model:
                #save checkpoint
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, f"{ckpt_folder}/ckpt_{global_step}.pt")

                if successes.mean() > best_success_rate:
                    best_success_rate = successes.mean()
                    torch.save({
                        'actor': actor.state_dict(),
                        'qf1': qf1_target.state_dict(),
                        'qf2': qf2_target.state_dict(),
                        'log_alpha': log_alpha,
                    }, f"{save_folder}/best_model.pt")

                    print(f"model saved to {save_folder}/best_model.pt with a success rate of {successes.mean()}")
                else:
                    print(f"model NOT saved due to the success rate of {successes.mean()} lower than {best_success_rate}")


        ## ROLLOUT (try not to modify: CRUCIAL step easy to overlook)
        rollout_time = time.time()
        old_obs, info = envs.reset_mm(seed=args.seed)
        obs, _, _, _, _ = envs.step_mm(envs.action_space.sample())

        state = process_obs_dict(obs, old_obs, (main_mode,), device)[0]
        old_obs = apply_noise(copy.deepcopy(old_obs), envs)
        obs = apply_noise(copy.deepcopy(obs), envs)
        obs_stack = process_obs_dict(obs, old_obs, args.modes, device)

        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = actor.get_train_action(obs_stack)
                actions = actions.detach()

            next_obs, rewards, terminations, truncations, infos = envs.step_mm(actions)
            real_next_obs = copy.deepcopy(next_obs)
            if args.bootstrap_at_done == 'always':
                next_done = torch.zeros_like(terminations).to(torch.float32)
            else:
                next_done = (terminations | truncations).to(torch.float32)

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k in real_next_obs:
                    real_next_obs[k][done_mask] = infos["final_observation"][k][done_mask]

                episodic_return = final_info['episode']['r'][done_mask].cpu().numpy().mean()
                if "success" in final_info:
                    writer.add_scalar("charts/success_rate", final_info["success"][done_mask].cpu().numpy().mean(),
                                      global_step)
                if "fail" in final_info:
                    writer.add_scalar("charts/fail_rate", final_info["fail"][done_mask].cpu().numpy().mean(),
                                      global_step)
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", final_info["elapsed_steps"][done_mask].cpu().numpy().mean(),
                                  global_step)

            next_state = process_obs_dict(next_obs, obs, (main_mode,), device)[0]
            real_next_obs = apply_noise(copy.deepcopy(real_next_obs), envs)
            real_next_obs_stack = process_obs_dict(real_next_obs, obs, args.modes, device)

            rb.add(state, obs_stack, next_state, real_next_obs_stack, actions, rewards, next_done)
            obs_stack = real_next_obs_stack

        rollout_time = time.time() - rollout_time

        ## UPDATING AGENT (ALGO LOGIC: training.)
        if global_step < args.learning_starts:
            continue
        update_time = time.time()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_train_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_states, next_state_actions)
                qf2_next_target = qf2_target(data.next_states, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf1_a_values = qf1(data.states, data.actions).view(-1)
            qf2_a_values = qf2(data.states, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_train_action(data.obs)
                qf1_pi = qf1(data.states, pi)
                qf2_pi = qf2(data.states, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_train_action(data.obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        update_time = time.time() - update_time

        ## LOGGING (Log training-related data)
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

            writer.add_scalar("losses/alpha", alpha, global_step)
            writer.add_scalar("charts/update_time", update_time, global_step)
            writer.add_scalar("charts/rollout_time", rollout_time, global_step)
            writer.add_scalar("charts/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    return (actor, qf1_target, qf2_target), log_alpha


def apply_noise(obs, envs):
    if not all(item in envs.obs_modes for item in obs.keys()):
        raise ValueError(f'The observation modes are not supported ({envs.obs_modes})')
    for m in obs.keys():
        space = envs.single_observation_space_mm.spaces[(envs.obs_modes.index(m))]
        mean = 0 if  m in ['rgb', 'segmentation'] else 0
        std = 10 if  m in ['rgb', 'segmentation'] else 1000
        obs[m] = torch.clip(
            obs[m] + torch.normal(mean, std, obs[m].shape).int().to(obs[m].device),
            min=float(space.low_repr), max=float(space.high_repr)
        )
    return obs

SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "rgb", "depth", "segmentation", "rgbd", "rgb+depth", "rgb+depth+segmentation", "rgb+segmentation", "depth+segmentation", "pointcloud")

if __name__ == "__main__":

    ## CONFIGS
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{'_'.join(sorted(args.modes.split('+')))}/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    writer = None
    if not args.evaluate:
        print("Running training\n")
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                group=args.wandb_group,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        print("Running evaluation\n")

    # TRY NOT TO MODIFY: seeding
    print(f"... seeding setup", end='\r')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ## ENVIRONMENTS SETUP
    print('... environments setup', end='\r')

    print('... building training environment', end='\r')
    eval_env_args = {
        'run_name':run_name,
        'game':'manipulation',
        'checkpoint_dir': os.path.join(os.getcwd(), 'test_runs'),
        'partial_reset': args.partial_reset,
        'noise_frequency': args.eval_noise_freq,
        'noise_types': args.eval_noise_types.split('+'),
        'eval_steps': args.num_eval_steps,
    }
    eval_envs = build_eval_env(
        id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        device="cuda" if torch.cuda.is_available() else "cpu",
        capture_video=args.capture_video,
        **eval_env_args
    )

    print('... building eval environment', end='\r')

    train_env_args = {
        'run_name':run_name,
        'game':'manipulation',
        'partial_reset': args.partial_reset,
        'noise_frequency': args.train_noise_freq,
        'noise_types': args.train_noise_types.split('+'),
    }

    envs = build_training_env(
        id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="rgb+depth+segmentation",#"state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        device="cuda" if torch.cuda.is_available() else "cpu",
        capture_video=args.capture_video,
        **train_env_args
    )

    print('... directories setup', end='\r')

    args.modes = sorted(args.modes.split("+"))
    save_folder = f"runs/{run_name}"
    ckpt_folder = f"{save_folder}/checkpoints"
    os.makedirs(ckpt_folder, exist_ok=True)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    print('... actor-critic setup', end='\r')

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(
        envs,
        modes=args.modes,
        fusion_strategy=args.fusion_strategy
    ).to(device)
    main_mode=args.modes[0]
    qf1 = SoftQNetwork(input_dim=envs.single_observation_space.spaces['sensor_data'][envs.data_source][main_mode],
                       output_dim = envs.single_action_space.shape[0], mode=main_mode).to(device)
    qf2 = SoftQNetwork(input_dim=envs.single_observation_space.spaces['sensor_data'][envs.data_source][main_mode],
                       output_dim = envs.single_action_space.shape[0], mode=main_mode).to(device)
    qf1_target = SoftQNetwork(input_dim=envs.single_observation_space.spaces['sensor_data'][envs.data_source][main_mode],
                       output_dim = envs.single_action_space.shape[0], mode=main_mode).to(device)
    qf2_target = SoftQNetwork(input_dim=envs.single_observation_space.spaces['sensor_data'][envs.data_source][main_mode],
                       output_dim = envs.single_action_space.shape[0], mode=main_mode).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha


    ## REPLAY BUFFER setup
    print('... replay buffer setup', end='\r')
    envs.single_observation_space_mm.dtype = np.float32
    rb = ReplayBuffer(
        state_shape=(2,*envs.single_observation_space.spaces['sensor_data'][envs.data_source][main_mode].shape[:2]),
        obs_shape=[envs.single_observation_space.spaces['sensor_data'][envs.data_source][mode].shape[:2] for mode in args.modes],
        act_shape=envs.single_action_space.shape,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=torch.device(device),
    )

    train_args = {
        'alpha':alpha
    }

    ## TRAINING
    print('... starting loop', end='\r')
    (actor, qf1_target, qf2_target), log_alpha = train(**train_args)
    envs.close()
