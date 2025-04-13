import argparse
import os

import torch
import yaml
from ignite.contrib import metrics

from skimage import measure
from sklearn.metrics import auc

import constants as const
import dataset
import fastflow
import utils

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
import time
from datetime import timedelta

#addded save visualization function calculate_aupro function and evaluate with visualization function

def save_visualization(image, anomaly_map, save_dir, filename, threshold=0.5):
    os.makedirs(save_dir, exist_ok=True)
    image_np = image.cpu().numpy()
    anomaly_map_np = anomaly_map.cpu().numpy()
    if image_np.shape[0] == 3:  # If RGB
        image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    anomaly_map_np = (anomaly_map_np - anomaly_map_np.min()) / (anomaly_map_np.max() - anomaly_map_np.min() + 1e-8)

    binary_mask = anomaly_map_np > threshold
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_normal.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    

    plt.figure(figsize=(10, 10))
    plt.imshow(anomaly_map_np, cmap='jet')
    plt.colorbar(label='Anomaly Score')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_heatmap.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    

    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.imshow(anomaly_map_np, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_overlay.png"), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Saved visualizations for {filename} to {save_dir}")
    
def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model):
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )
    return model

def evaluate_with_visualization(args, save_dir="./results", visualize=False):
    os.makedirs(save_dir, exist_ok=True)

    timing = {
        'loading': 0,
        'inference': 0,
        'metrics': 0
    }
    start_time = time.time() 
    stage_start = time.time()
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    if os.path.isdir(args.checkpoint):
        checkpoint_files = [f for f in os.listdir(args.checkpoint) if f.endswith('.pt') or f.endswith('.pth')]
        if len(checkpoint_files) == 0:
            raise ValueError(f"No checkpoint found in {args.checkpoint}")

        checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoint, x)), reverse=True)
        checkpoint_path = os.path.join(args.checkpoint, checkpoint_files[0])
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint

    checkpoint = torch.load(checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    

    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    model.cuda()
    model.eval()
 
    timing['loading'] = time.time() - stage_start
    stage_start = time.time()
    
    pixel_auroc_metric = metrics.ROC_AUC()
    
    image_scores = []
    image_labels = []

    all_pixel_scores = []
    all_pixel_labels = []
   
    pro_data = []  
    inference_times = []
    inference_times_by_status = {'normal': [], 'anomaly': []}
    for idx, (data, targets) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        data = data.cuda()

        inference_start = time.time()
        with torch.no_grad():
            ret = model(data)
        inference_end = time.time()
        inference_time = inference_end - inference_start
        inference_times.append(inference_time)

        anomaly_map = ret["anomaly_map"].cpu().detach()
        
        is_anomaly = torch.max(targets).item() > 0.5
        image_labels.append(1 if is_anomaly else 0)
        status = "anomaly" if is_anomaly else "normal"
    
        inference_times_by_status[status].append(inference_time)
 
        image_score = torch.max(anomaly_map).item()
        image_scores.append(image_score)
   
        anomaly_map_flat = anomaly_map.flatten()
        targets_flat = targets.flatten()

        all_pixel_scores.append(anomaly_map_flat.numpy())
        all_pixel_labels.append(targets_flat.numpy())

        if torch.min(targets_flat).item() < torch.max(targets_flat).item():
            pixel_auroc_metric.update((anomaly_map_flat, targets_flat))
            if is_anomaly:
                pro_data.append((anomaly_map.squeeze().numpy(), targets.squeeze().numpy()))

    timing['inference'] = time.time() - stage_start
    
    total_images = len(inference_times)
    normal_images = len(inference_times_by_status['normal'])
    anomaly_images = len(inference_times_by_status['anomaly'])
    
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    median_inference_time = np.median(inference_times) if inference_times else 0
    min_inference_time = np.min(inference_times) if inference_times else 0
    max_inference_time = np.max(inference_times) if inference_times else 0
    std_inference_time = np.std(inference_times) if inference_times else 0
    
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

    avg_normal_time = np.mean(inference_times_by_status['normal']) if inference_times_by_status['normal'] else 0
    avg_anomaly_time = np.mean(inference_times_by_status['anomaly']) if inference_times_by_status['anomaly'] else 0

    inference_stats = {
        'avg_inference_time': avg_inference_time,
        'median_inference_time': median_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'std_inference_time': std_inference_time,
        'fps': fps,
        'avg_normal_time': avg_normal_time,
        'avg_anomaly_time': avg_anomaly_time,
        'total_images': total_images,
        'normal_images': normal_images,
        'anomaly_images': anomaly_images
    }
    
    stage_start = time.time()

    if visualize:
        for i in range(data.size(0)):
            image = data[i].cpu()
            single_anomaly_map = anomaly_map[i].squeeze()

            filename = f"{args.category}_{idx:04d}_{status}"

            save_visualization(
                image=image,
                anomaly_map=single_anomaly_map,
                save_dir=save_dir,
                filename=filename
                )

    results = {}
    #everything in try except bc we ran into issues with the metrics not being able to compute
    #and we want to avoid crashing the entire evaluation process
    try:
        pixel_auroc = pixel_auroc_metric.compute()
        results["pixel_auroc"] = float(pixel_auroc)
    except Exception as e:
        print(f"Error computing pixel-level AUROC: {e}")
        results["pixel_auroc"] = 0.0

    try:
        image_auroc = roc_auc_score(image_labels, image_scores)
        results["image_auroc"] = float(image_auroc)
    except Exception as e:
        print(f"Error computing image-level AUROC: {e}")
        results["image_auroc"] = 0.0
 
    try:
        all_pixel_scores_flat = np.concatenate(all_pixel_scores)
        all_pixel_labels_flat = np.concatenate(all_pixel_labels)
        pixel_ap = average_precision_score(all_pixel_labels_flat, all_pixel_scores_flat)
        results["pixel_ap"] = float(pixel_ap)
    except Exception as e:
        print(f"Error computing pixel-level AP: {e}")
        results["pixel_ap"] = 0.0

    try:
        image_ap = average_precision_score(image_labels, image_scores)
        results["image_ap"] = float(image_ap)
    except Exception as e:
        print(f"Error computing image-level AP: {e}")
        results["image_ap"] = 0.0
    
    try:
        aupro = calculate_aupro(pro_data)
        results["aupro"] = float(aupro)
    except Exception as e:
        print(f"Error computing AUPRO: {e}")
        results["aupro"] = 0.0

    timing['metrics'] = time.time() - stage_start

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_formatted = str(timedelta(seconds=int(elapsed_time)))

    print("\n----- Evaluation Results -----")
    print(f"Category: {args.category}")
    print(f"Pixel-level AUROC: {results['pixel_auroc']:.4f}")
    print(f"Image-level AUROC: {results['image_auroc']:.4f}")
    print(f"Pixel-level AP: {results['pixel_ap']:.4f}")
    print(f"Image-level AP: {results['image_ap']:.4f}")
    print(f"AUPRO: {results['aupro']:.4f}")
   
    print("\n----- Inference Performance -----")
    print(f"Total images processed: {inference_stats['total_images']} ({inference_stats['normal_images']} normal, {inference_stats['anomaly_images']} anomaly)")
    print(f"Average inference time: {inference_stats['avg_inference_time']*1000:.2f} ms (± {inference_stats['std_inference_time']*1000:.2f} ms)")
    print(f"Median inference time: {inference_stats['median_inference_time']*1000:.2f} ms")
    print(f"Range: {inference_stats['min_inference_time']*1000:.2f} - {inference_stats['max_inference_time']*1000:.2f} ms")
    print(f"Throughput: {inference_stats['fps']:.2f} FPS (frames per second)")
    
    if inference_stats['normal_images'] > 0 and inference_stats['anomaly_images'] > 0:
        print(f"Normal images: {inference_stats['avg_normal_time']*1000:.2f} ms")
        print(f"Anomaly images: {inference_stats['avg_anomaly_time']*1000:.2f} ms")
        relative_diff = ((inference_stats['avg_anomaly_time'] / inference_stats['avg_normal_time']) - 1) * 100
        print(f"Difference: {relative_diff:.1f}% {'slower' if relative_diff > 0 else 'faster'} for anomaly images")
    
    print(f"\nTotal evaluation time: {elapsed_time_formatted} (hh:mm:ss)")
    print(f"  - Loading time: {timing['loading']:.2f}s ({timing['loading']/elapsed_time*100:.1f}%)")
    print(f"  - Inference time: {timing['inference']:.2f}s ({timing['inference']/elapsed_time*100:.1f}%)")
    print(f"  - Metrics calculation time: {timing['metrics']:.2f}s ({timing['metrics']/elapsed_time*100:.1f}%)")
    

    results['evaluation_time_seconds'] = elapsed_time
    results['loading_time_seconds'] = timing['loading']
    results['inference_time_seconds'] = timing['inference']
    results['metrics_calculation_seconds'] = timing['metrics']

    for key, value in inference_stats.items():
        results[key] = value

    with open(os.path.join(save_dir, "results.txt"), "w") as f:
        f.write(f"Category: {args.category}\n")
        for metric, value in results.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.4f}\n")
            else:
                f.write(f"{metric}: {value}\n")

        f.write(f"\nTiming Breakdown:\n")
        f.write(f"Total evaluation time: {elapsed_time_formatted}\n")
        f.write(f"Loading time: {timing['loading']:.2f}s ({timing['loading']/elapsed_time*100:.1f}%)\n")
        f.write(f"Inference time: {timing['inference']:.2f}s ({timing['inference']/elapsed_time*100:.1f}%)\n")
        f.write(f"Metrics calculation time: {timing['metrics']:.2f}s ({timing['metrics']/elapsed_time*100:.1f}%)\n")

        f.write(f"\nInference Performance:\n")
        f.write(f"Total images processed: {inference_stats['total_images']} ({inference_stats['normal_images']} normal, {inference_stats['anomaly_images']} anomaly)\n")
        f.write(f"Average inference time: {inference_stats['avg_inference_time']*1000:.2f} ms (± {inference_stats['std_inference_time']*1000:.2f} ms)\n")
        f.write(f"Median inference time: {inference_stats['median_inference_time']*1000:.2f} ms\n")
        f.write(f"Range: {inference_stats['min_inference_time']*1000:.2f} - {inference_stats['max_inference_time']*1000:.2f} ms\n")
        f.write(f"Throughput: {inference_stats['fps']:.2f} FPS (frames per second)\n")
        
        if inference_stats['normal_images'] > 0 and inference_stats['anomaly_images'] > 0:
            f.write(f"Normal images: {inference_stats['avg_normal_time']*1000:.2f} ms\n")
            f.write(f"Anomaly images: {inference_stats['avg_anomaly_time']*1000:.2f} ms\n")
            relative_diff = ((inference_stats['avg_anomaly_time'] / inference_stats['avg_normal_time']) - 1) * 100
            f.write(f"Difference: {relative_diff:.1f}% {'slower' if relative_diff > 0 else 'faster'} for anomaly images\n")

        if False: 
            f.write("\nPer-image inference times (ms):\n")
            for i, t in enumerate(inference_times):
                status = "anomaly" if image_labels[i] == 1 else "normal"
                f.write(f"Image {i+1} ({status}): {t*1000:.2f} ms\n")
    
    return results


def calculate_aupro(pro_data):

    
    max_steps = 100
    num_anomalous_images = 0
    total_pro = 0.0
    
    print(f"Calculating AUPRO for {len(pro_data)} anomalous images...")
    for i, (anomaly_map, mask) in enumerate(pro_data):
        if anomaly_map.max() > 1.0 or anomaly_map.min() < 0.0:
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-10)
    
        labeled_mask, num_components = measure.label(mask > 0.5, return_num=True)
        
        if num_components == 0:
            continue
            
        num_anomalous_images += 1
        print(f"  Image {i+1}/{len(pro_data)}: Found {num_components} anomalous regions")

        pro_curve = []
        thresholds = np.linspace(0, 1, max_steps)
        
        for threshold in thresholds:
            binary_map = anomaly_map >= threshold

            pro = 0.0
            for component_id in range(1, num_components + 1):
                region_mask = labeled_mask == component_id
                if np.sum(region_mask) == 0:
                    continue
    
                intersection = np.sum(binary_map & region_mask)
                overlap = intersection / np.sum(region_mask)
                pro += overlap
            pro = pro / num_components
            pro_curve.append(pro)
        image_aupro = auc(thresholds, pro_curve)
        total_pro += image_aupro
        print(f"  Image {i+1} AUPRO: {image_aupro:.4f}")

    if num_anomalous_images > 0:
        final_aupro = total_pro / num_anomalous_images
        print(f"Final AUPRO (averaged over {num_anomalous_images} images): {final_aupro:.4f}")
        return final_aupro
    else:
        print("Warning: No valid anomalous regions found for AUPRO calculation.")
        return 0.0
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)