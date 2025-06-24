import os
from stable_baselines3.common.callbacks import EvalCallback
import time
import numpy as np
import os


def is_truncated_from_infos(infos: list[dict]) -> bool:
    """
    Given a list of info dicts (one per env), return True if any of the dicts
    contain a key with 'truncated' in its name and that value is True.

    Parameters
    ----------
    infos : list[dict]
        List of info dictionaries returned from a VecEnv.

    Returns
    -------
    bool
        True if any truncated-related key is True in any info dict.
    """
    for info in infos:
        if any(key.lower().find("truncated") != -1 and info[key] for key in info):
            return True
    return False


class TimedEvalCallback(EvalCallback):
    """
    Custom callback function to also track training time expenses
    """

    def __init__(self, *args, time_log_path=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.eval_times = []
        self.time_log_path = time_log_path or os.path.join(
            self.log_path, "evaluation_times.npy"
        )

        os.makedirs(os.path.dirname(self.time_log_path), exist_ok=True)

    def _on_step(self) -> bool:
        start_time = time.time()
        result = super()._on_step()
        end_time = time.time()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            duration = end_time - start_time
            self.eval_times.append(duration)
            np.save(self.time_log_path, np.array(self.eval_times))

        return result
