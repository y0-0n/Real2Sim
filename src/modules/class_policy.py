import wandb, os, copy, ray, sys
import numpy as np
import torch 
import torch.nn as nn 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import *
from modules.class_dlpg import DeepLatentPolicyGradientClass
from modules.class_pid import PIDControllerClass
from modules.class_grp import GaussianRandomPathClass, scaleup_traj, get_anchors_from_traj
from modules.class_ray import RayRolloutWorkerClass

class AgentTrajectoryUpdateClass():
    def __init__(self,
                name = "CVAE Trajectory",
                env  = None,
                workers = None,
                target_env = None,
                leg_idx = 12345,
                env_ray = None,
                k_p  = 0.2,
                k_i  = 0.001,
                k_d  = 0.01,
                out_min = -2,
                out_max = +2, 
                ANTIWU  = True,
                z_dim    = 32,
                c_dim    = 3,
                h_dims   = [128, 128],
                var_max  = 0.1,
                n_anchor = 20,
                dur_sec  = 2,
                max_repeat    = 5,
                hyp_prior     = {'g': 1/1, 'l': 1/8, 'w': 1e-8},
                hyp_posterior = {'g': 1/4, 'l': 1/8, 'w': 1e-8},
                lbtw_base     = 0.8,
                device_idx = 0,
                VERBOSE    = True,
                WANDB      = False,
                SAVE_WEIGHTS = False,
                SAVE_PLOT    = False,
                SAVE_REWARD  = False,
                args       = None
                ):
        # Init params
        self.name       = name
        self.env        = env
        self.target_env = target_env(xml_path='xml/snapbot_4/robot_4_1245_heavy.xml', ctrl_coef=0.000, body_coef=0.000, vel_coef=0e-5, head_coef=0e-4, render_mode=None, VERBOSE=False)
        self.workers    = workers
        self.leg_idx    = leg_idx
        self.env_ray    = env_ray
        self.z_dim      = z_dim
        self.c_dim      = c_dim
        self.n_anchor   = n_anchor
        self.dur_sec    = dur_sec   
        self.max_repeat = max_repeat
        self.hyp_prior     = hyp_prior
        self.hyp_posterior = hyp_posterior
        self.lbtw_base   = lbtw_base
        self.VERBOSE = VERBOSE
        self.WANDB   = WANDB
        self.SAVE_WEIGHTS = SAVE_WEIGHTS
        self.SAVE_PLOT    = SAVE_PLOT
        self.SAVE_REWARD  = SAVE_REWARD
        self.args    = args
        self.RAY = True
        self.device = torch.device('cpu')
        # Set grp & pid & qscaler
        self.PID  = PIDControllerClass(name="PID", k_p=k_p, k_i=k_i, k_d=k_d, dim=self.env.adim, out_min=out_min, out_max=out_max, ANTIWU=ANTIWU)
        self.DLPG = DeepLatentPolicyGradientClass(name='DLPG', x_dim=env.adim*n_anchor, c_dim=c_dim, z_dim=z_dim, h_dims=h_dims, actv_enc=nn.ReLU(), actv_dec=nn.ReLU(), actv_q=nn.Softplus(), actv_out=None, var_max=var_max, device=self.device)
        self.QScaler      = ScalerClass(obs_dim=1)
        self.GRPPrior     = GaussianRandomPathClass(name='GRP Prior')
        self.GRPPosterior = GaussianRandomPathClass(name='GRP Posterior')
        # Set model to device
        self.DLPG.to(self.device)
        # Load weights for Real2Sim
        # self.DLPG.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../../real2sim/snapbot_33/6/0/weights/dlpg_model_weights_300.pth')))
        setattr(self.env, "condition", None)
        # Set explaination
        if self.VERBOSE:
            print("{} START with DEVICE: {}".format(self.name, self.device))

    def update(self,
                seed = 0,
                lr_dlpg     = 0.001,
                eps_dlpg    = 1e-8,
                n_worker    = 50,                
                epoch = 0,
                max_epoch   = 50,
                n_sim_roll          = 100,
                sim_update_size     = 64,
                n_sim_update        = 64,
                n_sim_prev_consider = 10,
                n_sim_prev_best_q   = 50,
                init_prior_prob = 0.5,
                folder = 0,
                idx = 0
                ):

        # Set buffer
        sim_x_list  = np.zeros((n_sim_roll, self.env.adim*self.n_anchor))
        sim_c_list  = np.zeros((n_sim_roll, self.c_dim))
        sim_q_list  = np.zeros(n_sim_roll)
        sim_x_lists = [''] * max_epoch
        sim_c_lists = [''] * max_epoch
        sim_q_lists = [''] * max_epoch
        eval_reward_list = []
        
        # Set margin
        ss_x_min  = -np.ones(self.env.adim)
        ss_x_max  = np.ones(self.env.adim)
        ss_margin = 0.1

        # Ray
        self.GRPPrior.set_prior(n_data_prior=4, dim=self.env.adim, dur_sec=self.dur_sec, HZ=self.env.hz, hyp=self.hyp_prior)
        traj_joints, traj_secs = self.GRPPrior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0) 
        t_anchor, _ = get_anchors_from_traj(traj_secs, traj_joints, n_anchor=self.n_anchor)

        while epoch < max_epoch:
            if idx != 0:
                train_rate = (epoch+300) / 500
            elif epoch / max_epoch < 1:
                train_rate = epoch / 500
            else:
                train_rate = 1
            exp_decrease_rate = 1 - train_rate  # 1 -> 0
            exp_increase_rate = 1 - exp_decrease_rate        # 0 -> 1
            prior_prob = init_prior_prob * exp_decrease_rate # Schedule eps-greedish (init_prior_prob -> 0)
            lbtw       = self.lbtw_base + (1-self.lbtw_base)*exp_increase_rate # Schedule leveraged GRP (0.8 -> 1.0)
            lbtw       = lbtw * 0.9 # max leverage to be 0.9

            for loop_idx in range(int(n_sim_roll/n_worker)):
                generate_trajectory_ray = [worker.generate_trajectory.remote(DLPG=self.DLPG, lbtw=lbtw, dur_sec=self.dur_sec, hyp_prior=self.hyp_prior, hyp_posterior=self.hyp_posterior, GRPPrior=self.GRPPrior, GRPPosterior=self.GRPPosterior, \
                                                                            ss_x_min=ss_x_min, ss_x_max=ss_x_max, ss_margin=ss_margin, prior_prob=prior_prob, epoch=epoch, n_anchor=self.n_anchor, t_anchor=t_anchor, traj_secs=traj_secs, idx=idx) for worker in self.workers]
                result_generate_trajectory = ray.get(generate_trajectory_ray)
                rollout_ray = [worker.rollout.remote(self.PID, result_generate_trajectory[i]['traj_joints_deg'], n_traj_repeat=self.max_repeat, RENDER=False) for i, worker in enumerate(self.workers)]
                result_rollout = ray.get(rollout_ray)
                for worker_idx in range(n_worker):
                    sim_x_list[worker_idx+loop_idx*n_worker, :] = np.copy(result_generate_trajectory[worker_idx]['x_anchor'].reshape(1, -1))
                    sim_c_list[worker_idx+loop_idx*n_worker, :] = np.copy(result_generate_trajectory[worker_idx]['c'])
                    sim_q_list[worker_idx+loop_idx*n_worker]    = np.copy(sum(result_rollout[worker_idx]['forward_rewards']))
            
            sim_x_lists[epoch] = copy.deepcopy(sim_x_list)
            sim_c_lists[epoch] = copy.deepcopy(sim_c_list)
            sim_q_lists[epoch] = copy.deepcopy(sim_q_list)

            for n_prev_idx in range(n_sim_prev_consider):
                if n_prev_idx == 0:
                    sim_x_list_bundle = sim_x_list
                    sim_c_list_bundle = sim_c_list
                    sim_q_list_bundle = sim_q_list
                else:
                    sim_x_list_bundle = np.concatenate((sim_x_list_bundle, sim_x_lists[max(0, epoch-n_prev_idx)]), axis=0)
                    sim_c_list_bundle = np.concatenate((sim_c_list_bundle, sim_c_lists[max(0, epoch-n_prev_idx)]), axis=0)
                    sim_q_list_bundle = np.concatenate((sim_q_list_bundle, sim_q_lists[max(0, epoch-n_prev_idx)]))
            
            sorted_idx  = np.argsort(-sim_q_list_bundle)
            sim_x_train = sim_x_list_bundle[sorted_idx[:n_sim_prev_best_q], :]
            sim_c_train = sim_c_list_bundle[sorted_idx[:n_sim_prev_best_q], :]
            sim_q_train = sim_q_list_bundle[sorted_idx[:n_sim_prev_best_q]]

            sim_x_train = np.concatenate((sim_x_train, sim_x_list), axis=0)
            sim_c_train = np.concatenate((sim_c_train, sim_c_list), axis=0)
            sim_q_train = np.concatenate((sim_q_train, sim_q_list))

            rand_idx    = np.random.permutation(sim_x_list_bundle.shape[0])[:n_sim_roll]
            sim_x_rand  = sim_x_list_bundle[rand_idx, :]
            sim_c_rand  = sim_c_list_bundle[rand_idx, :]
            sim_q_rand  = sim_q_list_bundle[rand_idx]
            sim_x_train = np.concatenate((sim_x_train, sim_x_rand), axis=0)
            sim_c_train = np.concatenate((sim_c_train, sim_c_rand), axis=0)
            sim_q_train = np.concatenate((sim_q_train, sim_q_rand))

            self.QScaler.reset()
            self.QScaler.update(sim_q_train)
            sim_q_scale, sim_q_offset = self.QScaler.get()
            sim_scaled_q = sim_q_scale * (sim_q_train-sim_q_offset)
            if epoch < 30 and idx != 0:
                recon_loss_gain = 1
                beta = 0.00001
            else:
                recon_loss_gain = 1
                beta = 0.01
            
            total_loss, recon_loss, kl_loss = self.DLPG.update(x=sim_x_train, c=sim_c_train, q=sim_scaled_q, lr=lr_dlpg, eps=eps_dlpg, 
                                                                recon_loss_gain=recon_loss_gain, beta=beta, max_iter=n_sim_update, batch_size=sim_update_size)

            # For eval
            c = [0,1,0]
            x_anchor = self.DLPG.sample_x(c=torch.FloatTensor(c).reshape(1,-1).to(self.device), n_sample=1, SKIP_Z_SAMPLE=True)[0].reshape(self.n_anchor, self.env.adim)
            x_anchor[-1,:] = x_anchor[0,:]
            self.GRPPosterior.set_posterior(t_anchor, x_anchor, lbtw=1.0, t_test=traj_secs, hyp=self.hyp_posterior, APPLY_EPSRU=True, t_eps=0.025)
            policy4eval_traj   = self.GRPPosterior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0, ss_x_min=ss_x_min, ss_x_max=ss_x_max, ss_margin=ss_margin)[0]
            policy4eval_traj   = scaleup_traj(self.env, policy4eval_traj, DO_SQUASH=True, squash_margin=5)
            t_anchor, x_anchor = get_anchors_from_traj(traj_secs, policy4eval_traj, n_anchor=self.n_anchor)  
            policy4eval  = rollout(self.env, self.PID, policy4eval_traj, n_traj_repeat=self.max_repeat)
            policy4eval_target  = rollout(self.target_env, self.PID, policy4eval_traj, n_traj_repeat=self.max_repeat)
            eval_secs    = policy4eval['secs']
            eval_xy_degs = policy4eval['xy_degs']
            eval_xy_degs_target = policy4eval_target['xy_degs']
            eval_reward  = sum(policy4eval['forward_rewards'])
            eval_x_diff  = policy4eval['x_diff']
            eval_reward_list.append(eval_reward)

            # For wandb
            if self.WANDB:
                wandb.log({"sim_reward": eval_reward, "sim_x_diff": eval_x_diff, "total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss})

            # Save model weights
            if self.SAVE_WEIGHTS:
                if (epoch+1) % 50 == 0 and epoch != 0:
                    if not os.path.exists("real2sim/{}/{}/{}/weights".format(folder, seed, idx)):
                        os.makedirs("real2sim/{}/{}/{}/weights".format(folder, seed, idx))
                    torch.save(self.DLPG.state_dict(), 'real2sim/{}/{}/{}/weights/dlpg_model_weights_{}.pth'.format(folder, seed, idx, epoch+1))

            # For printing evaluation of present policy
            if (epoch+1) % 5 == 0 and epoch != 0:
                print("[{:>3} / {}] Clear, CONDITION: [{}], REWARD: {:>6.2f}, XDIFF: {:>4.3f}".format(epoch+1, max_epoch, c, eval_reward, eval_x_diff))
            else:
                print("[{:>3} / {}] Clear".format(epoch+1, max_epoch))
            
            # Save Agent's trajectories
            
            if self.SAVE_PLOT:
                if (epoch+1) % 20 == 0 and epoch != 0:
                    if not os.path.exists("real2sim/{}/{}/{}/rviz".format(folder, seed, idx)):
                        os.makedirs("real2sim/{}/{}/{}/rviz".format(folder, seed, idx))
                    np.save('real2sim/{}/{}/{}/rviz/xy_degs_{}.npy'.format(folder, seed, idx, epoch+1), eval_xy_degs)
                    np.save('real2sim/{}/{}/{}/rviz/xy_degs_{}_target.npy'.format(folder, seed, idx, epoch+1), eval_xy_degs_target)


                    # if not os.path.exists("real2sim/{}/{}/{}/plot".format(folder, idx, seed)):
                    #     os.makedirs("real2sim/{}/{}/{}/plot".format(folder, idx, seed))
                    # plot_agent_joint_traj_and_topdown_traj(traj_secs, policy4eval_traj, t_anchor, x_anchor, eval_xy_degs, eval_secs,
                    #                                 figsize=(16,8), title_str='EPOCH: {:>3} REWARD: {:>6.2f} X_DIFF: {:.2f}'.format(epoch+1, eval_reward, eval_x_diff), 
                    #                                 tfs=15, SAVE=True, image_name='real2sim/{}/{}/{}/plot/epoch_{}.png'.format(folder, seed, idx, epoch+1))
            
                
            epoch += 1
            
        if self.SAVE_REWARD:
            eval_reward_np = np.array(eval_reward_list)
            if not os.path.exists("results/{}".format(folder)):
                os.makedirs("results/{}".format(folder))
            np.save('results/{}/{}/{}.npy'.format(folder, idx, seed), eval_reward_np)
                
        # ray.shutdown()

if __name__ == "__main__":
    env = AntRandomEnvClass(rand_mass=None, rand_fric=None, render_mode=None)
    env_ray = AntRandomEnvClass
    AgentTrajectoryUpdateClass = AgentTrajectoryUpdateClass(
                                                                name = "CVAE Trajectory",
                                                                env  = env,
                                                                leg_idx = 12345,
                                                                env_ray = env_ray,
                                                                k_p  = env.k_p,
                                                                k_i  = env.k_i,
                                                                k_d  = env.k_d,
                                                                out_min = -1,
                                                                out_max = +1, 
                                                                ANTIWU  = True,
                                                                z_dim    = 32,
                                                                c_dim    = 3,
                                                                h_dims   = [128, 128],
                                                                var_max  = -1,
                                                                n_anchor = 20,
                                                                dur_sec  = 2,
                                                                max_repeat    = 5,
                                                                hyp_prior     = {'g': 1/1, 'l': 1/8, 'w': 1e-8},
                                                                hyp_posterior = {'g': 1/4, 'l': 1/8, 'w': 1e-8},
                                                                lbtw_base     = 0.8,
                                                                device_idx = -1,
                                                                VERBOSE    = True,
                                                                WANDB      = False,
                                                                args       = None
                                                                )
    AgentTrajectoryUpdateClass.update(
                                        seed = 0,
                                        lr_dlpg     = 0.001,
                                        eps_dlpg    = 1e-8,
                                        n_worker    = 50,    
                                        epoch = 0,
                                        max_epoch   = 300,
                                        n_sim_roll          = 100,
                                        sim_update_size     = 64,
                                        n_sim_update        = 64,
                                        n_sim_prev_consider = 10,
                                        n_sim_prev_best_q   = 50,
                                        init_prior_prob = 0.5,
                                        folder = 43
                                        )
