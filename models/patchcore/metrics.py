import numpy as np
from sklearn import metrics

# added aupro and average precision 
def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    # Compute Average Precision Score (AUPRO for image-wise is just AP)
    ap = metrics.average_precision_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR, AUPRO) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)
    
    # Compute pixel-wise Average Precision
    ap = metrics.average_precision_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "ap": ap,
    }


def compute_pro_metric(anomaly_segmentations, ground_truth_masks, num_thresholds=100):

    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    ground_truth_masks = ground_truth_masks.astype(bool)
    
    max_anomaly_value = np.max(anomaly_segmentations)
    min_anomaly_value = np.min(anomaly_segmentations)

    thresholds = np.linspace(min_anomaly_value, max_anomaly_value, num_thresholds)
    
    pro_scores = []
    for i in range(len(anomaly_segmentations)):
        img_pro_scores = []

        from scipy.ndimage import label
        gt_labeled, num_regions = label(ground_truth_masks[i])
        if num_regions == 0:
            continue
        for threshold in thresholds:
            binary_anomaly = anomaly_segmentations[i] >= threshold
            region_overlaps = []
            for region_id in range(1, num_regions + 1):
                gt_region = gt_labeled == region_id
                gt_region_size = np.sum(gt_region)
                overlap = np.logical_and(binary_anomaly, gt_region)
                overlap_size = np.sum(overlap)

                if gt_region_size > 0:
                    region_overlap = overlap_size / gt_region_size
                    region_overlaps.append(region_overlap)
            if region_overlaps:
                img_pro_scores.append(np.mean(region_overlaps))
            else:
                img_pro_scores.append(0.0)
        pro_scores.append(img_pro_scores)

    avg_pro_scores = np.mean(pro_scores, axis=0) if pro_scores else np.zeros(num_thresholds)
    normalized_thresholds = np.linspace(0, 1, num_thresholds)
    aupro = np.trapz(avg_pro_scores, normalized_thresholds)
    
    return {
        "aupro": aupro,
        "pro_scores": avg_pro_scores,
        "thresholds": thresholds
    }