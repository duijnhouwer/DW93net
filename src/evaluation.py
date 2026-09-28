"""Shared float32 inference and float64 metric reduction for both evaluators."""
import math
import numpy as np
import torch

from features.reichardt import ReichardtBank
from models.flatland_flow import FlatlandFlowNet
from stimuli.load_flow_xt_data import make_flow_loader
from stimuli.metadata import data_contract
from precision import configure_tf32


METRIC_DEFINITIONS = {
    'normalized_mse': 'mean(sum((prediction-target)^2 / motion_scale^2)); validation loss drives LR and checkpoint selection.',
    'heading_mae_deg': 'Mean absolute wrapped heading error in degrees, in [0,180].',
    'heading_relative_mae_pct': '100 * heading_mae_deg / 180; not percent of true heading.',
    'heading_cosine_similarity': 'Mean normalized translation dot product; 1 aligned, 0 orthogonal, -1 opposite.',
    'heading_cosine_error_pct': 'Mean 50 * (1 - cosine similarity); 0 aligned, 50 orthogonal, 100 opposite.',
    'heading_within_15_pct': 'Percent of all samples within 15 degrees; undefined headings count as failures.',
    'speed_relative_mae_pct': 'Mean per-sample 100 * abs(predicted_speed - true_speed) / true_speed.',
    'vector_tip_relative_mae_pct': 'Mean per-sample 100 * norm(predicted_vector - true_vector) / true_speed.',
    'undefined_heading_count': 'Samples with predicted or true speed <= 1e-12; excluded from angular/cosine means.',
    'undefined_relative_error_count': 'Samples with true speed <= 1e-12; excluded from relative speed/vector means.',
    'speed_rmse': 'RMS speed error, in scene units per step.',
    'vector_rmse': 'RMS Euclidean translation vector error, in scene units per step.',
    'normalized_rmse': 'sqrt(mean(sum((prediction-target)^2 / motion_scale^2))) over samples.',
}


class NonfiniteEvaluationError(ValueError):
    """Numerical model failure during held-out inference."""


def inference_batches(checkpoint, dataset, *, device, batch_size, num_workers=0, indices=None):
    configure_tf32(checkpoint['allow_tp32'])
    if checkpoint['data_contract'] != data_contract(dataset.storage_metadata):
        raise ValueError('Checkpoint dataset geometry or generation implementation does not match')
    detector = checkpoint['detector']
    if detector['dog'] != dataset.metadata:
        raise ValueError('Checkpoint DoG configuration does not match dataset')
    bank = ReichardtBank(detector['correlation']['dx_values'])
    if bank.metadata != detector['correlation']:
        raise ValueError('Unsupported checkpoint correlation configuration')
    model = FlatlandFlowNet(bank.dx_values).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    subset = dataset if indices is None else torch.utils.data.Subset(dataset, indices)
    loader = make_flow_loader(subset, batch_size, device, num_workers=num_workers)
    with torch.inference_mode():
        for contrast, target in loader:
            prediction = model(bank(contrast.to(device, non_blocking=True).float()[:, None],
                                    validate=False)).float()
            if not torch.isfinite(target).all():
                raise ValueError('Nonfinite evaluation targets')
            if not torch.isfinite(prediction).all():
                raise NonfiniteEvaluationError('Nonfinite evaluation predictions')
            yield prediction.cpu().numpy(), target.numpy()


def summary_metrics(batches, scales):
    count = heading_count = 0
    squared = speed = tip = rotation = heading = 0.
    relative_count = within_15 = 0
    speed_pct = tip_pct = cosine = cosine_error = speed_squared = tip_squared = 0.
    for prediction, target in batches:
        errors = translation_errors(prediction[:, :2], target[:, :2])
        difference = prediction.astype(np.float64) - target
        squared += float(np.square(difference / np.asarray(scales)).sum())
        speed += float(errors['speed_absolute'].sum())
        tip += float(errors['tip_distance'].sum())
        rotation += float(np.abs(difference[:, 2]).sum())
        heading += float(errors['absolute'].sum())
        heading_count += len(errors['absolute'])
        relative_count += len(errors['relative_speed_error_pct'])
        within_15 += int((errors['absolute'] <= 15).sum())
        speed_pct += float(errors['relative_speed_error_pct'].sum())
        tip_pct += float(errors['relative_tip_error_pct'].sum())
        cosine += float(errors['cosine_similarity'].sum())
        cosine_error += float(errors['cosine_error_pct'].sum())
        speed_squared += float(np.square(errors['speed_absolute']).sum())
        tip_squared += float(np.square(errors['tip_distance']).sum())
        count += len(target)
    if not count:
        raise ValueError('Evaluation dataset is empty')
    result = dict(n_samples=count, normalized_mse=squared / count, normalized_rmse=math.sqrt(squared / count),
                  translation_speed_mae=speed / count, translation_vector_tip_mae=tip / count,
                  rotation_mae_deg=rotation / count,
                  heading_mae_deg=heading / heading_count if heading_count else None)
    result.update(
        heading_relative_mae_pct=100 * heading / (180 * heading_count) if heading_count else None,
        heading_cosine_similarity=cosine / heading_count if heading_count else None,
        heading_cosine_error_pct=cosine_error / heading_count if heading_count else None,
        heading_within_15_pct=100 * within_15 / count,
        undefined_heading_count=count - heading_count,
        undefined_relative_error_count=count - relative_count,
        speed_relative_mae_pct=speed_pct / relative_count if relative_count else None,
        vector_tip_relative_mae_pct=tip_pct / relative_count if relative_count else None,
        speed_rmse=math.sqrt(speed_squared / count), vector_rmse=math.sqrt(tip_squared / count))
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise NonfiniteEvaluationError('Nonfinite evaluation metrics')
    return result


def translation_errors(pred, target):
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError('Expected matching [N,2] translation vectors')
    if not len(pred) or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Empty or nonfinite predictions/targets')
    speed, truth = np.linalg.norm(pred, axis=1), np.linalg.norm(target, axis=1)
    valid = (speed > 1e-12) & (truth > 1e-12)
    delta = np.arctan2(pred[valid, 0], pred[valid, 1]) - np.arctan2(target[valid, 0], target[valid, 1])
    angles = np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))
    absolute = np.abs(angles)
    relative_heading_error_pct = 100 * absolute / 180
    tip_distance = np.linalg.norm(pred - target, axis=1)
    speed_absolute = np.abs(speed - truth)
    relative_valid = truth > 1e-12
    relative_speed_error_pct = 100 * speed_absolute[relative_valid] / truth[relative_valid]
    relative_tip_error_pct = 100 * tip_distance[relative_valid] / truth[relative_valid]
    cosine_similarity = np.clip(
        np.sum(pred[valid] * target[valid], axis=1) / (speed[valid] * truth[valid]), -1, 1)
    return dict(speed=speed, truth=truth, valid=valid, angles=angles, absolute=absolute,
                relative_heading_error_pct=relative_heading_error_pct,
                tip_distance=tip_distance, speed_absolute=speed_absolute,
                relative_speed_error_pct=relative_speed_error_pct,
                relative_tip_error_pct=relative_tip_error_pct,
                cosine_similarity=cosine_similarity, cosine_error_pct=50 * (1 - cosine_similarity),
                relative_valid=relative_valid)


def mean_ci95(values):
    """Normal-approximation 95% confidence interval for a sample mean."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    margin = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return max(0.0, mean - margin), mean + margin


def metrics(pred, target):
    """Compute angular, speed, and Euclidean translation-vector tip errors."""
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError('Expected matching [N,2] translation vectors')
    if not len(pred) or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Empty or nonfinite predictions/targets')
    errors = translation_errors(pred, target)
    speed, truth, valid = errors['speed'], errors['truth'], errors['valid']
    angles, absolute = errors['angles'], errors['absolute']
    relative_heading_error_pct = errors['relative_heading_error_pct']
    tip_distance, speed_absolute = errors['tip_distance'], errors['speed_absolute']
    relative_speed_error_pct = errors['relative_speed_error_pct']
    relative_tip_error_pct = errors['relative_tip_error_pct']
    cosine_similarity = errors['cosine_similarity']
    heading_ci_low, heading_ci_high = mean_ci95(absolute) if len(absolute) else (None, None)
    relative_heading_ci_low, relative_heading_ci_high = (
        mean_ci95(relative_heading_error_pct) if len(relative_heading_error_pct) else (None, None))
    speed_ci_low, speed_ci_high = mean_ci95(speed_absolute)
    tip_ci_low, tip_ci_high = mean_ci95(tip_distance)
    relative_speed_ci_low, relative_speed_ci_high = (
        mean_ci95(relative_speed_error_pct) if len(relative_speed_error_pct) else (None, None))
    relative_tip_ci_low, relative_tip_ci_high = (
        mean_ci95(relative_tip_error_pct) if len(relative_tip_error_pct) else (None, None))
    return {
        'n': len(pred), 'undefined_heading_count': int((~valid).sum()),
        'undefined_relative_error_count': int((~errors['relative_valid']).sum()),
        'heading_mae_deg': float(absolute.mean()) if len(angles) else None,
        'heading_mae_deg_ci95_low': heading_ci_low,
        'heading_mae_deg_ci95_high': heading_ci_high,
        'heading_relative_mae_pct': float(relative_heading_error_pct.mean()) if len(absolute) else None,
        'heading_relative_mae_pct_ci95_low': relative_heading_ci_low,
        'heading_relative_mae_pct_ci95_high': relative_heading_ci_high,
        'heading_median_deg': float(np.median(absolute)) if len(angles) else None,
        'heading_p90_deg': float(np.percentile(absolute, 90)) if len(angles) else None,
        'heading_bias_deg': float(angles.mean()) if len(angles) else None,
        'heading_cosine_similarity': float(cosine_similarity.mean()) if len(cosine_similarity) else None,
        'heading_cosine_error_pct': float(errors['cosine_error_pct'].mean()) if len(cosine_similarity) else None,
        # Undefined estimates count as failures in threshold accuracy.
        'heading_within_15_pct': float(100 * (absolute <= 15).sum() / len(pred)),
        'speed_mae': float(speed_absolute.mean()),
        'speed_mae_ci95_low': speed_ci_low,
        'speed_mae_ci95_high': speed_ci_high,
        'speed_relative_mae_pct': float(relative_speed_error_pct.mean()) if len(relative_speed_error_pct) else None,
        'speed_relative_mae_pct_ci95_low': relative_speed_ci_low,
        'speed_relative_mae_pct_ci95_high': relative_speed_ci_high,
        'speed_bias': float((speed - truth).mean()),
        'speed_rmse': float(np.sqrt(np.mean((speed - truth)**2))),
        # The physical error of the predicted vector tip, ||[dx,dz]-[dx_hat,dz_hat]||.
        'vector_tip_mae': float(tip_distance.mean()),
        'vector_tip_mae_ci95_low': tip_ci_low,
        'vector_tip_mae_ci95_high': tip_ci_high,
        'vector_tip_relative_mae_pct': float(relative_tip_error_pct.mean()) if len(relative_tip_error_pct) else None,
        'vector_tip_relative_mae_pct_ci95_low': relative_tip_ci_low,
        'vector_tip_relative_mae_pct_ci95_high': relative_tip_ci_high,
        'vector_tip_median': float(np.median(tip_distance)),
        'vector_tip_p90': float(np.percentile(tip_distance, 90)),
        'vector_rmse': float(np.sqrt(np.mean(tip_distance**2))),
        'true_speed_min': float(truth.min()), 'true_speed_max': float(truth.max()),
    }


CARDINAL_HEADINGS = {0: 'Front', 90: 'Right', 180: 'Back', 270: 'Left'}


def empty_metrics():
    """Metric schema for a requested stratum that contains no samples."""
    return {
        'n': 0, 'undefined_heading_count': 0, 'undefined_relative_error_count': 0,
        'heading_cosine_error_pct': None,
        'heading_mae_deg': None, 'heading_median_deg': None,
        'heading_mae_deg_ci95_low': None, 'heading_mae_deg_ci95_high': None,
        'heading_relative_mae_pct': None, 'heading_relative_mae_pct_ci95_low': None,
        'heading_relative_mae_pct_ci95_high': None,
        'heading_p90_deg': None, 'heading_bias_deg': None, 'heading_cosine_similarity': None,
        'heading_within_15_pct': None, 'speed_mae': None,
        'speed_mae_ci95_low': None, 'speed_mae_ci95_high': None,
        'speed_relative_mae_pct': None, 'speed_relative_mae_pct_ci95_low': None,
        'speed_relative_mae_pct_ci95_high': None,
        'speed_bias': None, 'speed_rmse': None, 'vector_tip_mae': None,
        'vector_tip_mae_ci95_low': None, 'vector_tip_mae_ci95_high': None,
        'vector_tip_relative_mae_pct': None, 'vector_tip_relative_mae_pct_ci95_low': None,
        'vector_tip_relative_mae_pct_ci95_high': None,
        'vector_tip_median': None, 'vector_tip_p90': None, 'vector_rmse': None,
        'true_speed_min': None, 'true_speed_max': None,
    }


def heading_bin_label(center_deg, count):
    """Name a truth-heading bin; zero degrees is front and +90 is right."""
    center_deg = int(round(center_deg)) % 360
    if count == 1:
        return 'All headings'
    if count == 4:
        return CARDINAL_HEADINGS[center_deg]
    return f'{center_deg}°'


def binned_metrics(pred, target, *, dimension, bin_counts):
    """Return equal-width truth-heading or truth-speed metric strata.

    Heading bins are circular and centred at zero/front. Speed bins span the
    minimum and maximum true speeds in this fixed evaluation sample.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    if dimension == 'heading':
        values = np.mod(np.degrees(np.arctan2(target[:, 0], target[:, 1])), 360)
        span = 360.0
    elif dimension == 'speed':
        values = np.linalg.norm(target, axis=1)
        span = None
    else:
        raise ValueError(f'Unknown dimension: {dimension}')

    result = {}
    for count in bin_counts:
        if dimension == 'heading':
            width = span / count
            indices = np.floor(np.mod(values + width / 2, span) / width).astype(int)
            indices = np.minimum(indices, count - 1)
            bins = []
            for index in range(count):
                center = index * width
                lower = (center - width / 2) % span
                upper = (center + width / 2) % span
                mask = indices == index
                bins.append({
                    'label': heading_bin_label(center, count),
                    'center_deg': float(center), 'lower_deg': float(lower),
                    'upper_deg': float(upper), 'wraps': bool(lower > upper),
                    'metrics': metrics(pred[mask], target[mask]) if mask.any() else empty_metrics(),
                })
        else:
            low, high = float(values.min()), float(values.max())
            edges = np.linspace(low, high, count + 1)
            indices = np.minimum(np.searchsorted(edges, values, side='right') - 1, count - 1)
            bins = []
            for index in range(count):
                mask = indices == index
                bins.append({
                    'label': 'All speeds' if count == 1 else f'Speed {index + 1}',
                    'lower_speed': float(edges[index]), 'upper_speed': float(edges[index + 1]),
                    'upper_inclusive': index == count - 1,
                    'metrics': metrics(pred[mask], target[mask]) if mask.any() else empty_metrics(),
                })
        result[str(count)] = bins
    return result
