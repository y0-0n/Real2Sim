import torch, glob, os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import matplotlib.image as mpimg
from envs.snapbot.class_snapbot import Snapbot6EnvClass, Snapbot5EnvClass, Snapbot4EnvClass, Snapbot3EnvClass
from modules.class_policy import AgentTrajectoryUpdateClass
from modules.class_grp import *
from utils.utils import *

def eval_agent_from_network(env, dur_sec, n_anchor, max_repeat, folder, seed, epoch,  condition, RENDER=False, PLOT=True):
    EvalPolicy = AgentTrajectoryUpdateClass(
                                                name = "EvalPolicy",
                                                env  = env,
                                                k_p  = env.k_p,
                                                k_i  = env.k_i,
                                                k_d  = env.k_d,
                                                out_min = -1.0,
                                                out_max = +1.0, 
                                                ANTIWU  = True,
                                                z_dim    = 32,
                                                c_dim    = 3,
                                                h_dims   = [128, 128],
                                                var_max  = None,
                                                n_anchor = n_anchor,
                                                dur_sec  = dur_sec,
                                                max_repeat    = max_repeat,
                                                hyp_prior     = {'g': 1/1, 'l': 1/8, 'w': 1e-8},
                                                hyp_posterior = {'g': 1/4, 'l': 1/8, 'w': 1e-8},
                                                lbtw_base     = 0.8,
                                                device_idx = 0
                                                )
    try:
        EvalPolicy.DLPG.load_state_dict(torch.load("real2sim/{}/{}/weights/dlpg_model_weights_{}.pth".format(folder, seed, epoch), map_location='cuda:0'))
    except:
        EvalPolicy.DLPG.load_state_dict(torch.load("real2sim/{}/{}/weights/dlpg_model_weights_{}.pth".format(folder, seed, epoch), map_location='cpu'))
    EvalPolicy.DLPG.eval()
    EvalPolicy.GRPPrior.set_prior(n_data_prior=4, dim=env.adim, dur_sec=dur_sec, HZ=env.hz, hyp=EvalPolicy.hyp_prior)
    traj_joints, traj_secs = EvalPolicy.GRPPrior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0)
    t_anchor, x_anchor = get_anchors_from_traj(traj_secs, traj_joints, n_anchor=EvalPolicy.n_anchor) 
    ss_x_min  = -np.ones(env.adim)
    ss_x_max  = np.ones(env.adim)
    ss_margin = 0.1
    n_sample = 1
    for i in range(n_sample):
        x_anchor = EvalPolicy.DLPG.sample_x(c=torch.FloatTensor(condition).reshape(1,-1).to(EvalPolicy.device), n_sample=1, SKIP_Z_SAMPLE=True).reshape(EvalPolicy.n_anchor, EvalPolicy.env.adim)
        x_anchor[-1,:] = x_anchor[0,:]
        EvalPolicy.GRPPosterior.set_posterior(t_anchor,x_anchor,lbtw=1.0,t_test=traj_secs,hyp=EvalPolicy.hyp_posterior,APPLY_EPSRU=True,t_eps=0.025)
        # EvalPolicy.GRPPosterior.plot(n_sample=1,figsize=(15,5),subplot_rc=(2,4),lw_sample=1/2,tfs=10,rand_type='Uniform')
        policy4eval_traj, traj_secs = EvalPolicy.GRPPosterior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0, ss_x_min=ss_x_min, ss_x_max=ss_x_max, ss_margin=ss_margin)
        policy4eval_traj = scaleup_traj(EvalPolicy.env, policy4eval_traj, DO_SQUASH=False)
        policy4eval = rollout(EvalPolicy.env, EvalPolicy.PID, policy4eval_traj, n_traj_repeat=EvalPolicy.max_repeat, RENDER=RENDER, PLOT=PLOT)
        eval_reward = sum(policy4eval['forward_rewards'])
        eval_x_diff = policy4eval['x_diff']
        eval_figure = policy4eval['figure']
        if PLOT:
            eval_figure.savefig('for_plot_{}'.format(i))
        plt.close()
        print("REWARD: {:>.1f} X_DIFF: {:>.3f}".format(eval_reward, eval_x_diff))
    fig  = plt.figure()
    rows = n_sample
    cols = 1
    i = 1
    for idx, filename in enumerate(glob.glob("*.png")):
        img = mpimg.imread(filename)
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img)
        plt.axis('off')
        i += 1
        os.remove(filename)
        if idx == n_sample-1:
            break
    plt.show()
    
    return policy4eval_traj

if  __name__ == "__main__":
    # env = AntRandomEnvClass(rand_mass=None, rand_fric=None, render_mode=None)
    # env = AntRandomEnvClass(rand_mass=[1,4], rand_fric=None, render_mode=None)
    # env.set_leg_weight(4)
    # env = AntRandomEnvClassWithBox(rand_mass=[1, 4], rand_fric=None, render_mode=None) 
    # env = HalfCheetahRandomEnvClassWithBox(rand_mass=[1, 4], rand_fric=None, render_mode=None)
    # env = HalfCheetahRandomEnvClass(rand_mass=[1,4], rand_fric=None, render_mode=None)
    # env = HalfCheetahThreeLegsEnvClass(render_mode=None)
    # env = AntThreeLegsEnvClass(render_mode=None)
    # env.set_leg_weight(2) 
    # env = Snapbot6EnvClass(render_mode=None)
    # env = Snapbot4EnvClass(ctrl_coef=0.000, body_coef=0.000, vel_coef=0e-2, head_coef=0e-4, render_mode=None)
    env = Snapbot4EnvClass(ctrl_coef=0.000, body_coef=0.000, vel_coef=0e-5, head_coef=0e-4, xml_path='xml/snapbot_4/robot_4_1245_0.xml', render_mode=None)
    # env  = Snapbot6EnvClass(ctrl_coef=0.000, body_coef=0.000, vel_coef=0e-2, head_coef=0e-4, render_mode=None)
    traj = eval_agent_from_network(env=env, dur_sec=2, n_anchor=20, max_repeat=10, folder="snapbot_33", seed=6, epoch=50,  condition=[0,1,0], RENDER=True, PLOT=False)
    # eval_agent_from_network(env=env, dur_sec=2, n_anchor=20, max_repeat=5, folder="snapbot_5_s_12345_transfer", seed=1, epoch=300,  condition=[0,1,0], RENDER=True, PLOT=True)
    np.save('traj_5_5.npy', traj)