import copy
import os
import random
import time

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro
#from models import SoftQNetwork, Actor
from models import ActorMultimodal as Actor
from models import SoftQNetworkMultimodal as SoftQNetwork

import mani_skill.envs

from utils import Args, ReplayBufferSample, get_distance
from utils import ReplayBufferMultimodal as ReplayBuffer
from enviroment import EnvMultimodalWrapper
from process import image_preprocess
from collections import deque



def train(**kwargs):
    '''
    Needs:
     + models: (actor, qf1_target, qf2_target)
     + ReplayBuffer
    Returns:

    '''
    envs.reset_mm(seed=args.seed)
    eval_envs.reset_mm(seed=args.seed)

    global_step = 0
    global_update = 0
    learning_has_started = False
    alpha = kwargs['alpha']
    global_steps_per_iteration = args.num_envs * (args.steps_per_env)

    ## GLOBAL LOOP
    while global_step < args.total_timesteps:
        print(f"Global Step: {global_step}", end='\r')

        ## EVALUATION and SAVING
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            actor.eval()
            print("Evaluating")
            (old_eval_img, eval_obs), _ = eval_envs.reset_mm(seed=args.seed)
            old_action = torch.zeros(eval_envs.action_space.shape)
            (eval_img, eval_obs), _, _, _, _ = eval_envs.step_mm(eval_envs.action_space.sample())

            z_img, z_obs = actor.get_encodings((
                torch.cat([image_preprocess(old_eval_img), image_preprocess(eval_img)], dim=1).float(),
                eval_obs
            ))
            z_old = (z_img+z_obs)/2

            returns = []
            eps_lens = []
            successes = []
            failures = []
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    state = (torch.cat([image_preprocess(old_eval_img), image_preprocess(eval_img)], dim=1).float(), eval_obs)
                    action, z_old = actor.get_eval_action(state, z_old, old_action.to(device))
                    (next_eval_img, eval_obs), _, eval_terminations, eval_truncations, eval_infos = eval_envs.step_mm(action)
                    old_eval_img = eval_img.clone()
                    eval_img = next_eval_img.clone()
                    old_action = action.clone()

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
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")



        ## ROLLOUT (try not to modify: CRUCIAL step easy to overlook)
        rollout_time = time.time()
        (old_img, obs), info = envs.reset_mm(seed=args.seed)
        (img, obs), _, _, _, _ = envs.step_mm(envs.action_space.sample())

        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            img_stack = torch.cat([image_preprocess(old_img), image_preprocess(img)], dim=1)

            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                state = (img_stack, obs)
                actions, _, _ = actor.get_train_action(state)
                actions = actions.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            (next_img, next_obs), rewards, terminations, truncations, infos = envs.step_mm(actions)
            real_next_obs = next_obs.clone()
            next_img_stack = torch.cat([image_preprocess(img), image_preprocess(next_img)], dim=1)

            if args.bootstrap_at_done == 'always':
                next_done = torch.zeros_like(terminations).to(torch.float32)
            else:
                next_done = (terminations | truncations).to(torch.float32)
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[done_mask] = infos["final_observation"][done_mask]
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

            rb.add(obs, img_stack, real_next_obs, next_img_stack, actions, rewards, next_done)

            obs = next_obs
            old_img = img.clone()
            img = next_img.clone()

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
                next_state_actions, next_state_log_pi, _ = actor.get_train_action((data.next_img,data.next_obs))
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_train_action((data.img,data.obs))
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                # update the encoders
                (z_img_0, z_snd_0), z_0, (z_img_1, z_snd_1), z_1, z_act_1 = actor.get_representations(data)

                MM_loss = (torch.mean((get_distance(z_img_0, z_snd_0))) + torch.mean((get_distance(z_img_1, z_snd_1))))
                TF_loss = torch.mean(get_distance(z_act_1, z_1))
                TC_loss = torch.mean((get_distance(z_0, z_1) - 1.0) ** 2)
                rnd_idx = np.arange(z_0.shape[0])
                np.random.shuffle(rnd_idx)
                NTC_loss = - torch.mean(torch.log(get_distance(z_0, z_1[rnd_idx]) + 1e-6))
                representation_loss = args.TC_coeff * TC_loss + args.NTC_coeff * NTC_loss + args.TF_coeff * TF_loss + args.MM_coeff * MM_loss

                loss = actor_loss + representation_loss

                actor_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_train_action((data.img,data.obs))
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
            writer.add_scalar("losses/representation_loss", representation_loss.item(), global_step)
            writer.add_scalar("losses/multimodal_loss", MM_loss.item(), global_step)
            writer.add_scalar("losses/transfer_loss", TF_loss.item(), global_step)
            writer.add_scalar("losses/contrastive_positive_loss", TC_loss.item(), global_step)
            writer.add_scalar("losses/contrastive_negative_loss", NTC_loss.item(), global_step)

            writer.add_scalar("losses/alpha", alpha, global_step)
            writer.add_scalar("charts/update_time", update_time, global_step)
            writer.add_scalar("charts/rollout_time", rollout_time, global_step)
            writer.add_scalar("charts/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    return (actor, qf1_target, qf2_target), log_alpha



if __name__ == "__main__":

    ## CONFIGS
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    writer = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
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
        print("Running evaluation")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ## ENVIRONMENTS SETUP
    env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", sim_backend="gpu")
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, **env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.partial_reset, **env_kwargs)

    envs = EnvMultimodalWrapper(envs)
    eval_envs = EnvMultimodalWrapper(eval_envs)


    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
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
    envs.single_observation_space_mm.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=torch.device(device)
    )

    train_args = {
        'alpha':alpha
    }

    ## TRAINING
    (actor, qf1_target, qf2_target), log_alpha = train(**train_args)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()
    envs.close()
