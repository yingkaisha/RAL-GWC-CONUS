import logging
from credit.losses.base_losses import base_losses
from credit.losses.weighted_loss import VariableTotalLoss2D


logger = logging.getLogger(__name__)


def load_loss(conf, reduction="none", validation=False):
    
    loss_conf = conf["loss"]
    use_weighted_loss = loss_conf.get("use_latitude_weights", False) or loss_conf.get("use_variable_weights", False)

    if use_weighted_loss:
        logger.info("Weighted loss applied")
        return VariableTotalLoss2D(conf, validation=validation)
        
    return base_losses(conf, reduction=reduction, validation=validation)
