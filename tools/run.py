"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import argparse

import numpy as np

import gym
from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.sac.twin_sac import TwinSAC
from rlkit.torch.networks import FlattenMlp


parser = argparse.ArgumentParser(description='Run RL using RLKIT')
parser.add_argument('--env-name', type=str, default='Ant-v2')
parser.add_argument('--exp-name', type=str, default='baseline')
parser.add_argument('--algo', type=str, default='sac')  # support 'sac', 'tsac'
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--cpu', default=False, action='store_true')
args = parser.parse_args()


def experiment(variant):
    env = NormalizedBoxEnv(gym.make(args.env_name))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    net_size = variant['net_size']

    algo_map = dict(
        sac=dict(
            algo=SoftActorCritic,
            network=dict(
                policy=TanhGaussianPolicy(
                    hidden_sizes=[net_size, net_size],
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                ),
                qf=FlattenMlp(
                    hidden_sizes=[net_size, net_size],
                    input_size=obs_dim + action_dim,
                    output_size=1,
                ),
                vf=FlattenMlp(
                    hidden_sizes=[net_size, net_size],
                    input_size=obs_dim,
                    output_size=1,
                ),
            )
        ),
        tsac=dict(
            algo=TwinSAC,
            network=dict(
                policy=TanhGaussianPolicy(
                    hidden_sizes=[net_size, net_size],
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                ),
                qf1=FlattenMlp(
                    hidden_sizes=[net_size, net_size],
                    input_size=obs_dim + action_dim,
                    output_size=1,
                ),
                qf2=FlattenMlp(
                    hidden_sizes=[net_size, net_size],
                    input_size=obs_dim + action_dim,
                    output_size=1,
                ),
                vf=FlattenMlp(
                    hidden_sizes=[net_size, net_size],
                    input_size=obs_dim,
                    output_size=1,
                ),
            )
        ),
    )

    algo_type = algo_map[args.algo]
    algorithm = algo_type['algo'](
        env=env,
        **algo_type['network'],
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def main():
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,
            reward_scale=1,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
    )
    setup_logger(args.env_name, variant=variant, exp_id=args.exp_name)
    ptu.set_gpu_mode(not args.cpu, gpu_id=args.gpu_id)
    experiment(variant)


if __name__ == "__main__":
    main()
