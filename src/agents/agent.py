""" Unifying interface for both Demonstrator and Learner classes as they share some functionalities"""

from abc import ABC
from environments.environment import Environment

class Agent(ABC):
    """
    Unifying class for both Demonstrator and Learner classes as they share some functionalities
    
    Parameters
    ----------
    env : environment.Environment
        the environment representing the setting of the problem
    agent_name : str
        name of the agent
    """
    def __init__(self, env:Environment, agent_name:str):
        self.env = env
        self.agent_name = agent_name
        self.policy = None
        self.solver=None
        self.V = None
        self.T = None


    def draw(self, show: bool = False, store: bool = False, fignum: int = 0) -> None:
        """
        draws the policy of the demonstrator as long as it has been computed before, else a warning is thrown

        Parameters
        ----------
        show : bool
            whether or not the plot should be shown
        store : bool
            whether or not the plot should be stored
        fignum : int
            identifier number for the figure
        """

        
        self.env.render(
            V=self.V,
            policy=self.policy,
            reward=self.env.reward,
            show=show,
            strname=self.agent_name,
            fignum=fignum,
            store=store,
            T=self.T,
        )
    