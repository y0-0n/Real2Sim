import argparse, json, sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from envs.snapbot.class_snapbot import Snapbot6EnvClass, Snapbot5EnvClass, Snapbot4EnvClass, Snapbot3EnvClass
from modules.class_policy import AgentTrajectoryUpdateClass
from modules.class_bayesian_optimization import BayesianOptimizationClass
from modules.class_ray import RayRolloutWorkerClass
import ray, torch, wandb
import numpy as np
# from typing import Namespace
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        frac = float(num) / float(denom)
        return frac
        
def main(args:argparse.Namespace):
    env = Snapbot4EnvClass(ctrl_coef=0.000, body_coef=0.000, vel_coef=0e-5, head_coef=0e-4, render_mode=None)
    target_env = Snapbot4EnvClass
    env_ray = Snapbot4EnvClass

    args.hyp_prior = json.loads(args.hyp_prior)
    args.hyp_posterior = json.loads(args.hyp_posterior)
    for k, v in args.hyp_prior.items():
        args.hyp_prior[k] = convert_to_float(v)
    for k, v in args.hyp_posterior.items():
        args.hyp_posterior[k] = convert_to_float(v)
    # Set random seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set wandb 
    wandb.init(project="real2sim", entity="l5vd5", name=str(args.folder)+"-real2sim-"+str(args.seed))
    wandb.config.update(args)
    
    ray.init(num_cpus=args.n_worker)
    workers = [RayRolloutWorkerClass.remote(env=env_ray, leg_idx=args.leg_idx, rand_mass=env.rand_mass, device=torch.device('cpu'), worker_id=i) for i in range(int(args.n_worker))]
    target_workers = RayRolloutWorkerClass.remote(env=target_env, leg_idx=args.leg_idx, rand_mass=env.rand_mass, device=torch.device('cpu'), worker_id=-1)

    AgentPolicy = AgentTrajectoryUpdateClass(
                                            name = "CVAE Trajectory",
                                            env  = env,
                                            workers = workers,
                                            leg_idx = args.leg_idx,
                                            env_ray = env_ray,
                                            k_p  = env.k_p,
                                            k_i  = env.k_i,
                                            k_d  = env.k_d,
                                            out_min = -args.torque,
                                            out_max = +args.torque, 
                                            ANTIWU  = True,
                                            z_dim    = args.z_dim,
                                            c_dim    = args.c_dim,
                                            h_dims   = args.h_dims,
                                            var_max  = args.var_max,
                                            n_anchor = args.n_anchor,
                                            dur_sec  = args.dur_sec,
                                            max_repeat    = args.max_repeat,
                                            hyp_prior     = args.hyp_prior,
                                            hyp_posterior = args.hyp_posterior,
                                            lbtw_base     = args.lbtw_base,
                                            device_idx = args.device_idx,
                                            VERBOSE    = args.VERBOSE,
                                            WANDB = args.WANDB,
                                            SAVE_WEIGHTS = args.SAVE_WEIGHTS,
                                            SAVE_PLOT    = args.SAVE_PLOT,
                                            SAVE_REWARD  = args.SAVE_REWARD,
                                            args  = args                                           
                                            )
    bayesian_optim = BayesianOptimizationClass(
                        policy=AgentPolicy,
                        target_workers=target_workers,
                        # env=env,
                        # env_ray=env_ray,
                        n_worker=args.n_worker,
                        workers = workers,
                        # target_env=target_env,
                    )
    # initial mass
    init_mass = bayesian_optim.x_sampler(1)[0][0][0]
    

    for loop_idx in range(args.n_loop):
        if loop_idx == 0:
            [worker.change_parameter.remote(mass=init_mass) for worker in workers]
        else:
            [worker.change_parameter.remote(mass=x_sol[0][0]) for worker in workers]
        AgentPolicy.update(
                            seed = args.seed,
                            lr_dlpg     = args.lr_dlpg,
                            eps_dlpg    = args.eps_dlpg,
                            n_worker    = args.n_worker,                        
                            start_epoch = args.start_epoch,
                            max_epoch   = args.max_epoch,
                            n_sim_roll          = args.n_sim_roll,
                            sim_update_size     = args.sim_update_size,
                            n_sim_update        = args.n_sim_update,
                            n_sim_prev_consider = args.n_sim_prev_consider,
                            n_sim_prev_best_q   = args.n_sim_prev_best_q,
                            init_prior_prob = args.init_prior_prob,
                            folder = args.folder,
                            loop_idx = loop_idx
                            )
        x_sol, y_sol = bayesian_optim.optim()
        print("reward gap : {}".format(y_sol))
        print("current mass : {}".format(x_sol))
        print("target mass : {}".format(2.243375116892044))
        
        

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="snapbot_6", choices=["ant_standard", "ant_leg", "ant_box", "ant_three_leg",
                                                                  "half_cheetah_standard", "half_cheetah_leg", "half_cheetah_box", "half_cheetah_three_leg",
                                                                  "snapbot_6", "snapbot_5", "snapbot_4", "snapbot_3"])
    parser.add_argument("--leg_idx", default="12345", type=str)                                                                                                                                                                                  
    parser.add_argument("--torque", default=1, help="setting for max torque", type=float)
    parser.add_argument("--z_dim", default=32, type=int)
    parser.add_argument("--c_dim", default=3, type=int)
    parser.add_argument("--h_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--var_max", default=-1, type=float)
    parser.add_argument("--n_anchor", default=20, type=int)
    parser.add_argument("--dur_sec", default=2, type=float)
    parser.add_argument("--max_repeat", default=5, type=int)
    parser.add_argument("--hyp_prior", default='{"g":"1/1", "l":"1/8", "w":"1e-8"}', type=str)
    parser.add_argument("--hyp_posterior", default='{"g": "1/4", "l": "1/8", "w": "1e-8"}', type=str)
    parser.add_argument("--lbtw_base", default=0.8, type=float)
    parser.add_argument("--device_idx", default=0, type=int)
    parser.add_argument("--VERBOSE", default=True, type=str2bool)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lr_dlpg", default=0.001, type=float)
    parser.add_argument("--eps_dlpg", default=1e-8, type=float)
    parser.add_argument("--n_worker", default=50, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--max_epoch", default=300, type=int)
    parser.add_argument("--n_sim_roll", default=100, type=int)
    parser.add_argument("--sim_update_size", default=64, type=int)
    parser.add_argument("--n_sim_update", default=64, type=int)
    parser.add_argument("--n_sim_prev_consider", default=10, type=int)
    parser.add_argument("--n_loop", default=100, type=int)
    parser.add_argument("--n_sim_prev_best_q", default=50, type=int)
    parser.add_argument("--init_prior_prob", default=0.8, type=float)
    parser.add_argument("--folder", default=0, type=str)
    parser.add_argument("--WANDB", default=False, type=str2bool)
    parser.add_argument("--SAVE_WEIGHTS", default=False, type=str2bool)
    parser.add_argument("--SAVE_PLOT", default=False, type=str2bool)
    parser.add_argument("--SAVE_REWARD", default=False, type=str2bool)
    args = parser.parse_args()
    main(args)
