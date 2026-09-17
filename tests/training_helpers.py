"""Deterministic held-out metrics for isolated training-engine unit tests."""
from train.train_flow_xt_net import train_recipe_set as train_engine


def fixed_validation(checkpoint):
    return dict(normalized_mse=1., normalized_rmse=1., n_samples=2,
                heading_mae_deg=0., translation_speed_mae=0.,
                translation_vector_tip_mae=0., rotation_mae_deg=0.)


def train_recipe_set(cfg, **kwargs):
    kwargs.setdefault('validate_epoch', fixed_validation)
    return train_engine(cfg, **kwargs)
