import MDPSolver2
import copy

class Demonstrator:
    def __init__(self, env, myname):
        #self.Q = None
        self.V = None
        self.pi = None
        self.reward = None
        self.env = env
        self.myname = myname
        self.mu_demonstrator = self.get_mu_usingRewardFeatures(self.env, self.env.reward)

    def get_mu_usingRewardFeatures(self, env, reward):
        Q, V, pi_d, pi_s = MDPSolver2.MDPSolver.valueIteration(env, dict(reward=reward))
        self.V = V
        self.pi = pi_s
        _, mu, nu = MDPSolver2.MDPSolver.computeFeatureSVF_bellmann_averaged(env, pi_s)
        return mu[:env.n_features_reward], nu[:env.n_features_reward,:env.n_features_reward]


    def compute_and_draw(self, fignum, paper=False):
        """
        The flag "paper" indicates whether plots for the paper should be generated or not.
        """
        self.reward = copy.deepcopy(self.env.reward)
        #Q_teacher, V_teacher, _, pi_teacher = MDPSolver.valueIteration(self.env, self.env.reward)
        #self.Q = Q_teacher
        #self.V = V_teacher
        #self.pi = pi_teacher
        if paper:
            self.env.draw_paper(self.V, self.pi, self.reward, self.myname)
        else:
            self.env.draw(self.V, self.pi, self.reward, False, self.myname, fignum)