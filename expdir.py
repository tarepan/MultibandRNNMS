from pathlib import Path
from typing import Tuple

def makeExpDirs(results_dir_path: str, exp_name: str)->Path:
    """Make experiment directory, which contain yielded results 
    /exp_name
        /params  
            (e.g.) model.ckpt-80000.pt  
        /logs  
            (e.g.) events.out.tfevents.1111  
        /samples  
            (e.g.) gen_xxx.wav
    """
    results_dir = Path(results_dir_path)
    exp_dir = results_dir/exp_name
    params = exp_dir/"params"
    logs = exp_dir/"logs"
    samples = exp_dir/"samples"
    for dir_path in [params, logs, samples]:
        dir_path.mkdir(parents=True, exist_ok=True)
    return exp_dir

if __name__ == '__main__':
    makeExpDirs("results", "trial2")