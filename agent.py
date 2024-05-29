import MDPSolver2
import numpy as np
import copy
_largenum = 1000000

class Agent:
    def __init__(self, env, mu_demonstrator, config_agent, myname, solver):
        self.env = env
        self.mu_demonstrator = mu_demonstrator
        self.theta_e = np.zeros(self.env.n_features_reward)
        self.theta_v = np.zeros((self.env.n_features_reward,self.env.n_features_reward))
        
        self.tol = config_agent["tol"]
        self.maxiter = config_agent["maxiter"]
        self.miniter = config_agent["miniter"]

        self.theta_e_upperBound = _largenum
        self.theta_v_upperBound = _largenum

        #self.theta_e_gradientTerm = 1/config_agent["featReward_softconst_l2C"]
        self.theta_e_gradientTerm = 1/np.inf
        self.theta_v_gradientTerm = 1/np.inf
        
        #self.lambdas_pref_gradientTerm = config_agent["featPref_hardconst_eps"]
        #self.lambdas_pref_gradientTerm = 1e-09

        self.V = None
        self.pi = None
        self.reward = None
        self.solver = solver

        self.myname = myname

    def compute_and_draw(self, fignum, paper=False):
        self.reward = self.get_reward_for_given_thetas()
        self.variance = self.get_variance_for_given_thetas()
        Q_agent, V_agent, pi_agent = self.solver.soft_valueIteration(self.env, dict(reward=self.reward, variance=self.variance))
        self.pi = pi_agent
        #self.V = V_agent
        self.V = self.solver.computeValueFunction_bellmann_averaged(self.env, self.pi, dict(reward=self.env.reward, variance=self.env.variance)) # this is value of agent's policy w.r.t. env's reward
        if paper:
            print("XXXX", self.myname)
            self.env.draw_paper(self.V, self.pi, self.reward, self.myname)
        else:
            self.env.draw(self.V, self.pi, self.reward, False, self.myname, fignum)

    def get_reward_for_given_thetas(self):
        #w_reward = self.theta_e_pos - self.theta_e_neg
        w_reward = self.theta_e
        reward = self.env.get_reward_for_given_theta(w_reward)
        return reward
    
    def get_variance_for_given_thetas(self):
        #w_reward = self.theta_e_pos - self.theta_e_neg
        variance = self.theta_v
        variance = self.env.get_variance_for_given_theta(variance)
        return variance

    def get_mu_soft(self):
        reward_agent = self.get_reward_for_given_thetas()
        variance_agent = self.get_variance_for_given_thetas()
        Q, V, pi_s = self.solver.soft_valueIteration(self.env, dict(reward=reward_agent, variance=variance_agent))
        _, mu, nu = self.solver.computeFeatureSVF_bellmann_averaged(self.env, pi_s)
        return mu[:self.env.n_features_reward], nu[:self.env.n_features_reward, :self.env.n_features_reward], mu, nu


    def batch_MCE(self, verbose=True):

        calc_theta_v = isinstance(self.solver, MDPSolver2.MDPSolverVariance)

        theta_e_pos = np.zeros(self.env.n_features_reward)
        theta_e_neg = np.zeros(self.env.n_features_reward)
        self.theta_e = theta_e_pos - theta_e_neg

        if calc_theta_v:
            theta_v_pos = np.zeros((self.env.n_features_reward, self.env.n_features_reward))
            theta_v_neg = np.zeros((self.env.n_features_reward, self.env.n_features_reward))
            self.theta_v = theta_v_pos - theta_v_neg

        mu_reward_agent, mu_variance_agent, _,_ = self.get_mu_soft()

        if (verbose):
            print("\n========== batch_MCE for " + self.myname + " =======")
        #print(mu_agent)
        gradientconstant = 1
        t = 1
        while True:
            # set learning rate
            eta = gradientconstant / np.sqrt(t)
            #if(t > 100):
            #    eta = gradientconstant / np.sqrt(100)

            if(verbose):
                print("t=", t)
                print("...eta=", eta)
                print("...mu_reward_agent=", mu_reward_agent, " mu_demonstrator=", self.mu_demonstrator)
                print("...local_theta_e_pos=", theta_e_pos)
                print("...local_theta_e_neg=", theta_e_neg)
                print("...theta_e=", self.theta_e)
            
            # update lambda
            theta_e_pos_old = copy.deepcopy(theta_e_pos)
            theta_e_neg_old = copy.deepcopy(theta_e_neg)
            theta_e_old = copy.deepcopy(self.theta_e)

            if calc_theta_v:
                theta_v_pos_old = copy.deepcopy(theta_v_pos)
                theta_v_neg_old = copy.deepcopy(theta_v_neg)
                theta_v_old = copy.deepcopy(self.theta_v)

            theta_e_pos = theta_e_pos_old - eta * (mu_reward_agent - self.mu_demonstrator[0] + theta_e_pos_old*(self.theta_e_gradientTerm/2))
            theta_e_neg = theta_e_neg_old - eta * (self.mu_demonstrator[0] - mu_reward_agent + theta_e_neg_old*(self.theta_e_gradientTerm/2))

            if calc_theta_v:
                theta_v_pos = theta_v_pos_old - eta * (mu_variance_agent - self.mu_demonstrator[1] + theta_v_pos_old*(self.theta_v_gradientTerm/2))
                theta_v_neg = theta_v_neg_old - eta * (self.mu_demonstrator[1] - mu_variance_agent + theta_v_neg_old*(self.theta_v_gradientTerm/2))

        
            theta_e_pos = np.maximum(theta_e_pos, 0)
            theta_e_neg = np.maximum(theta_e_neg, 0)
            theta_e_pos = np.minimum(theta_e_pos, self.theta_e_upperBound)
            theta_e_neg = np.minimum(theta_e_neg, self.theta_e_upperBound)
            self.theta_e = theta_e_pos - theta_e_neg

            if calc_theta_v:
                theta_v_pos = np.maximum(theta_v_pos, 0)
                theta_v_neg = np.maximum(theta_v_neg, 0)
                theta_v_pos = np.minimum(theta_v_pos, self.theta_v_upperBound)
                theta_v_neg = np.minimum(theta_v_neg, self.theta_v_upperBound)
                self.theta_v = theta_v_pos - theta_v_neg

            # update state
            mu_reward_agent, mu_variance_agent, _, _ = self.get_mu_soft()

            diff_L2_norm_theta_e_pos = np.linalg.norm(theta_e_pos_old - theta_e_pos)
            diff_L2_norm_theta_e_neg = np.linalg.norm(theta_e_neg_old - theta_e_neg)
            diff_L2_norm_theta_e = np.linalg.norm(theta_e_old - self.theta_e)

            if calc_theta_v:
                #diff_L2_norm_theta_v_pos = np.linalg.norm(theta_v_pos_old - theta_v_pos)
                #diff_L2_norm_theta_v_neg = np.linalg.norm(theta_v_neg_old - theta_v_neg)
                diff_L2_norm_theta_v = np.linalg.norm(theta_v_old - self.theta_v)

            if(verbose):
                print("...diff_L2_norm_theta_e_pos=", diff_L2_norm_theta_e_pos)
                print("...diff_L2_norm_theta_e_neg=", diff_L2_norm_theta_e_neg)
                print("...diff_L2_norm_theta_e=", diff_L2_norm_theta_e)

            if calc_theta_v:
                if ((diff_L2_norm_theta_e  < self.tol) and (diff_L2_norm_theta_v < self.tol)):
                    if(t >= self.miniter):
                        break
            elif (diff_L2_norm_theta_e  < self.tol):
                if(t >= self.miniter):
                    break

            if (t > self.maxiter):
                break

            t += 1
    

