from typing import Callable, TypeVar, Optional
from dataclasses import dataclass

from omegaconf import OmegaConf, SCMode, MISSING

from .train import ConfTrain


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
data:
    batch_size: 32
    num_workers: null
    pin_memory: null
    adress_data_root: null
train:
    ckpt_log:
        dir_root: logs
        name_exp: default
        name_version: version_-1
    trainer:
        max_epochs: 500
        val_interval_epoch: 4
        profiler: null
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

@dataclass
class ConfData:
    """Configuration of data.
    Args:
        batch_size: Number of datum in a batch
        num_workers: Number of data loader worker
        pin_memory: Use data loader pin_memory
        adress_data_root: Root adress of data
    """
    batch_size: int = MISSING
    num_workers: Optional[int] = MISSING
    pin_memory: Optional[bool] = MISSING
    adress_data_root: Optional[str] = MISSING

@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        seed: PyTorch-Lightning's seed for every random system
        path_extend_conf: Path of configuration yaml which extends default config
    """
    seed: int = MISSING
    path_extend_conf: Optional[str] = MISSING
    data: ConfData = ConfData()
    train: ConfTrain = ConfTrain()


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
            conf_final = OmegaConf.merge(default, extends, cli)
        else:
            conf_final = OmegaConf.merge(default, cli)

        # Design Note -- OmegaConf instance v.s. DataClass instance --
        #   OmegaConf instance has runtime overhead in exchange for type safety.
        #   Configuration is constructed/finalized in early stage,
        #   so config is eternally valid after validation in last step of early stage.
        #   As a result, we can safely convert OmegaConf to DataClass after final validation.
        #   This prevent (unnecessary) runtime overhead in later stage.
        #
        #   One demerit: No "freeze" mechanism in instantiated dataclass.
        #   If OmegaConf, we have `OmegaConf.set_readonly(conf_final, True)`

        # [todo]: Return both dataclass and OmegaConf because OmegaConf has export-related utils.

        # `.to_container()` with `SCMode.INSTANTIATE` resolve interpolations and check MISSING.
        # It is equal to whole validation.
        return OmegaConf.to_container(conf_final, structured_config_mode=SCMode.INSTANTIATE)

    return generated_load_conf

load_conf = gen_load_conf(conf_default)
"""Load configuration type-safely.
"""
