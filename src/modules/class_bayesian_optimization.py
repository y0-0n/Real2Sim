import numpy as np
import matplotlib.pyplot as plt
import ray
import time
from utils.BO import acquisition_function
from modules.class_ray import RayRolloutWorkerClass
import torch
from modules.class_grp import GaussianRandomPathClass, scaleup_traj, get_anchors_from_traj

class BayesianOptimizationClass(object):
    def __init__(self,
                policy,
                workers,
                target_workers,
                x_minmax=np.array([[0.1, 5.0]]),
                # n_sample=100,
                n_random=1,
                n_bo=3,
                n_cd=0,
                n_worker=50,
                n_sample_max=1000
                ) -> None:
        self.x_minmax = x_minmax
        # self.n_sample = n_sample
        self.n_random = n_random
        self.n_bo = n_bo
        self.n_cd = n_cd
        self.n_worker = n_worker
        self.policy = policy
        self.target_workers = target_workers
        self.workers = workers
        self.n_sample_max = n_sample_max

    def x_sampler(self, n_sample):
        """
        Sample x as a list from the input domain 
        """
        x_samples = []
        for _ in range(n_sample):
            x_samples.append(
                self.x_minmax[:,0]+(self.x_minmax[:,1]-self.x_minmax[:,0])*np.random.rand(1,self.x_minmax.shape[0]))
        return x_samples

    def get_best_xy(self,x_data,y_data):
        """
        Get the current best solution
        """
        min_idx = np.argmin(y_data)
        return x_data[min_idx,:].reshape((1,-1)),y_data[min_idx,:].reshape((1,-1))

    def eval(self, result_rollout):
        target_result_rollout = result_rollout[-1]
        sample_result_rollout = result_rollout[:-1]
        # print(target_result_rollout)
        # print(np.tile(target_result_rollout, (len(sample_result_rollout),1)).shape)
        # target_result_rollout_tile = np.tile(target_result_rollout, (len(sample_result_rollout),1)).shape
        sample_x_diff = []
        for result in sample_result_rollout:
            sample_x_diff.append(result['x_diff'])
            # print(result['x_diff'])
        # print(target_result_rollout['x_diff'])
        sample_x_diff = np.array(sample_x_diff)
        target_x_diff = np.tile(target_result_rollout['x_diff'], len(sample_result_rollout))
        loss = np.sqrt([[(sample_x_diff-target_x_diff)**2]]).reshape(-1, 1, 1)
        return loss
    def get_best_xy(self,x_data,y_data):
        """
        Get the current best solution
        """
        min_idx = np.argmin(y_data)
        return x_data[min_idx,:].reshape((1,-1)),y_data[min_idx,:].reshape((1,-1))

    def optim(self):
        x_evals = self.x_sampler(self.n_worker) # mass candidate

        # GRP parameter
        self.policy.GRPPrior.set_prior(n_data_prior=4, dim=self.policy.env.adim, dur_sec=self.policy.dur_sec, HZ=self.policy.env.hz, hyp=self.policy.hyp_prior)
        traj_joints, traj_secs = self.policy.GRPPrior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0) 
        t_anchor, _ = get_anchors_from_traj(traj_secs, traj_joints, n_anchor=self.policy.n_anchor)

        # Set margin
        ss_x_min  = -np.ones(self.policy.env.adim)
        ss_x_max  = np.ones(self.policy.env.adim)
        ss_margin = 0.1

        # Change each simulator parameter
        [worker.change_parameter.remote(mass=x_evals[idx][0][0]) for idx, worker in enumerate(self.workers)]

        # Generate one trajectory (for target env, real-world)
        generate_trajectory = self.workers[0].generate_trajectory.remote(DLPG=self.policy.DLPG, lbtw=1.0, dur_sec=self.policy.dur_sec, hyp_prior=self.policy.hyp_prior, hyp_posterior=self.policy.hyp_posterior, GRPPrior=self.policy.GRPPrior, GRPPosterior=self.policy.GRPPosterior, \
                                                                    ss_x_min=ss_x_min, ss_x_max=ss_x_max, ss_margin=ss_margin, prior_prob=0., start_epoch=1, n_anchor=self.policy.n_anchor, t_anchor=t_anchor, traj_secs=traj_secs)
        result_trajectory = ray.get(generate_trajectory)
        workers_with_target = self.workers + [self.target_workers]
        rollout_ray = [worker.rollout.remote(self.policy.PID, result_trajectory['traj_joints_deg'], n_traj_repeat=self.policy.max_repeat, RENDER=False) for i, worker in enumerate(workers_with_target)]
        result_rollout = ray.get(rollout_ray)
        y_evals = self.eval(result_rollout)

        x_data,y_data = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]

        while True:
            for rd_idx in range(self.n_random):
                x_evals = self.x_sampler(self.n_worker) # mass candidate
                # Change each simulator parameter
                [worker.change_parameter.remote(mass=x_evals[idx][0][0]) for idx, worker in enumerate(self.workers)]

                # Generate one trajectory (for target env, real-world)
                generate_trajectory = self.workers[0].generate_trajectory.remote(DLPG=self.policy.DLPG, lbtw=1.0, dur_sec=self.policy.dur_sec, hyp_prior=self.policy.hyp_prior, hyp_posterior=self.policy.hyp_posterior, GRPPrior=self.policy.GRPPrior, GRPPosterior=self.policy.GRPPosterior, \
                                                                            ss_x_min=ss_x_min, ss_x_max=ss_x_max, ss_margin=ss_margin, prior_prob=0., start_epoch=1, n_anchor=self.policy.n_anchor, t_anchor=t_anchor, traj_secs=traj_secs)
                result_trajectory = ray.get(generate_trajectory)
                workers_with_target = self.workers + [self.target_workers]
                rollout_ray = [worker.rollout.remote(self.policy.PID, result_trajectory['traj_joints_deg'], n_traj_repeat=self.policy.max_repeat, RENDER=False) for i, worker in enumerate(workers_with_target)]
                result_rollout = ray.get(rollout_ray)
                y_evals = self.eval(result_rollout)
                x_random,y_random = np.asarray(x_evals)[:,0,:],np.asarray(y_evals)[:,0,:]
                x_data = np.concatenate((x_data,x_random),axis=0)
                y_data = np.concatenate((y_data,y_random),axis=0)

            print("x_data.shape[0]", x_data.shape[0])
            if x_data.shape[0] >= self.n_sample_max: break
            for bo_idx in range(self.n_bo):
                # Constant liar model for parallelizing BO
                x_evals,x_data_copy,y_data_copy = [],np.copy(x_data),np.copy(y_data)
                for _ in range(self.n_worker):
                    x_checks = np.asarray(self.x_sampler(n_sample=2000))[:,0,:]
                    a_ei,mu_checks,_ = acquisition_function(
                        x_data_copy,y_data_copy,x_checks,gain=1.0,invlen=5.0,eps=1e-6)
                    max_idx = np.argmax(a_ei)
                    x_liar,y_liar = x_checks[max_idx,:].reshape((1,-1)),mu_checks[max_idx].reshape((1,-1))
                    # Append
                    x_data_copy = np.concatenate((x_data_copy,x_liar),axis=0)
                    y_data_copy = np.concatenate((y_data_copy,y_liar),axis=0)
                    x_evals.append(x_liar)

                # Change each simulator parameter
                [worker.change_parameter.remote(mass=x_evals[idx][0][0]) for idx, worker in enumerate(self.workers)]

                # Generate one trajectory (for target env, real-world)
                generate_trajectory = self.workers[0].generate_trajectory.remote(DLPG=self.policy.DLPG, lbtw=1.0, dur_sec=self.policy.dur_sec, hyp_prior=self.policy.hyp_prior, hyp_posterior=self.policy.hyp_posterior, GRPPrior=self.policy.GRPPrior, GRPPosterior=self.policy.GRPPosterior, \
                                                                            ss_x_min=ss_x_min, ss_x_max=ss_x_max, ss_margin=ss_margin, prior_prob=0., start_epoch=1, n_anchor=self.policy.n_anchor, t_anchor=t_anchor, traj_secs=traj_secs)
                result_trajectory = ray.get(generate_trajectory)
                workers_with_target = self.workers + [self.target_workers]
                rollout_ray = [worker.rollout.remote(self.policy.PID, result_trajectory['traj_joints_deg'], n_traj_repeat=self.policy.max_repeat, RENDER=False) for i, worker in enumerate(workers_with_target)]
                result_rollout = ray.get(rollout_ray)
                y_evals = self.eval(result_rollout)
                x_bo = np.asarray(x_evals)[:,0,:]
                y_bo = np.asarray(y_evals)[:,0,:]
                x_data = np.concatenate((x_data,x_bo),axis=0)
                y_data = np.concatenate((y_data,y_bo),axis=0)
            if x_data.shape[0] >= self.n_sample_max: break
        x_sol,y_sol = self.get_best_xy(x_data,y_data)
        return x_sol, y_sol

# if __name__ == "__main__":

