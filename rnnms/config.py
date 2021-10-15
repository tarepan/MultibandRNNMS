from typing import Callable, TypeVar
from omegaconf import OmegaConf

from .main_train import ConfGlobal


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: None
data:
    batch_size: 32
    num_workers: None
    pin_memory: None
    adress_data_root: None
train:
    ckptLog:
        dir_root: logs
        name_exp: default
        name_version: version_-1
    trainer:
        max_epochs: 500
        val_interval_epoch: 4
        profiler: None
    model:
        sampling_rate: 16000
        vocoder:
            size_mel_freq: 80
            size_latent: 128
            bits_mu_law: 10
            hop_length: 200
            wave_ar:
                # size_i_cnd: local sync
                size_i_embed_ar: 256
                size_h_rnn: 896
                size_h_fc: 1024
                # size_o_bit: local sync
        optim:
            learning_rate: 4.0 * 1e-4
            sched_decay_rate: 0.5
            sched_decay_step: 25000
"""

def conf_default() -> ConfGlobal:
    """Default global configuration.
    """
    return OmegaConf.merge(
        OmegaConf.structured(ConfGlobal),
        OmegaConf.create(CONF_DEFAULT_STR)
    )

T = TypeVar('T')
def gen_load_conf(gen_conf_default: Callable[[], T], ) -> Callable[[], T]:
    """Generate 'Load configuration type-safely' function.

    Priority: CLI args > CLI-specified config yaml > Default

    Args:
        gen_conf_default: Function which generate default structured config
    """

    def generated_load_conf() -> T:
        default = gen_conf_default()
        cli = OmegaConf.from_cli()
        extends_path = cli.get("path_extend_conf", None)
        if extends_path:
            extends = OmegaConf.load(extends_path)
            return OmegaConf.merge(default, extends, cli)
        else:
            return OmegaConf.merge(default, cli)

    return generated_load_conf

load_conf = gen_load_conf(conf_default)
"""Load configuration type-safely.
"""
