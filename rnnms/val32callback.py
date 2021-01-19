from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import AMPType


class Val32Callback(Callback):
    """Validation-only no-AMP mode change.

    If no-AMP in default, there is no affect, don't worry!
    """

    def __init__(self) -> None:
        super().__init__()
        
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Any):
        trainer.amp_backend_val_switching = False

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Trun OFF native AMP temporally during validation.
        """

        if trainer.amp_backend is AMPType.NATIVE:
            print("Validation-only FP32.")
            trainer.amp_backend_val_switching = True
            trainer.amp_backend = None
        else:
            pass

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Trun ON back native AMP.
        """

        if trainer.amp_backend_val_switching:
            print("Back to native AMP.")
            trainer.amp_backend = AMPType.NATIVE
        else:
            pass
