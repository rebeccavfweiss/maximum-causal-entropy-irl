import os
from stable_baselines3.common.callbacks import EvalCallback
import time
import numpy as np
from scipy.sparse import diags


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

def laplacian_2d(H, W, F):
    N = H * W
    diagonals = []
    offsets = []

    # Main diagonal (degree)
    main_diag = F * np.ones(N)
    # edges have fewer neighbors
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if i == 0 or i == H - 1:
                main_diag[idx] -= 1
            if j == 0 or j == W - 1:
                main_diag[idx] -= 1
    diagonals.append(main_diag)
    offsets.append(0)

    # Neighboring pixels
    diagonals.extend([-1 * np.ones(N-1), -1 * np.ones(N-1), -1 * np.ones(N-W), -1 * np.ones(N-W)])
    offsets.extend([-1, 1, -W, W])

    L = diags(diagonals, offsets, shape=(N, N))
    return L
