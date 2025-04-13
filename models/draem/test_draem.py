import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import torch, torch.nn.functional as F, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, auc
import os, datetime, time, cv2
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# added aupro calc, visualization, and timing
def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(dataloader), size=(16,))


        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)


            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0]
                display_gt_images[cnt_display] = gray_batch[0]
                display_out_masks[cnt_display] = t_mask[0]
                display_in_masks[cnt_display] = true_mask[0]
                cnt_display += 1


            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("==============================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)
    
    
def calculate_aupro(gt_masks, pred_masks, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    pro_curve = np.zeros(num_thresholds)
    
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        if gt_mask.max() == 0: continue
        gt_binary = (gt_mask > 0).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(gt_binary)
        
        for label in range(1, num_labels):
            region_mask = (labels == label).astype(np.uint8)
            region_size = np.sum(region_mask)
            
            for i, threshold in enumerate(thresholds):
                binary_pred = (pred_mask >= threshold).astype(np.uint8)
                intersection = np.logical_and(binary_pred, region_mask).sum()
                pro = intersection / region_size if region_size > 0 else 0
                pro_curve[i] += pro
    
    num_regions = sum(cv2.connectedComponents((gt > 0).astype(np.uint8))[0] - 1 for gt in gt_masks)
    if num_regions > 0: pro_curve = pro_curve / num_regions
    
    return auc(thresholds, pro_curve), pro_curve, thresholds

def evaluate_draem(obj_name, visualize=False):
    img_dim, gpu_id = 256, 0
    mvtec_path ="../..\data\paperclips"
    checkpoint_path = "./ch"
    base_model_name = "DRAEM_test_0.001_700_bs4"
    run_name = f"{base_model_name}_{obj_name}_"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./outputs", f"{obj_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    timing = {'loading': 0, 'inference': 0, 'visualization': 0, 'metrics': 0}
    stage_start = time.time()

    torch.cuda.set_device(gpu_id)
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + ".pckl"), map_location='cuda:0'))
    model.cuda().eval()
    
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + "_seg.pckl"), map_location='cuda:0'))
    model_seg.cuda().eval()

    dataset = MVTecDRAEMTestDataset(mvtec_path +"/"+ obj_name + "/test/", resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    timing['loading'] = time.time() - stage_start

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt, anomaly_score_gt, anomaly_score_prediction = 0, [], []
    all_gt_masks, all_pred_masks = [], []
    reconstruction_times, segmentation_times, total_inference_times = [], [], []
    inference_times_by_status = {'normal': [], 'anomaly': []}

    stage_start = time.time()
    for i_batch, sample_batched in enumerate(tqdm(dataloader, desc=f"Processing {obj_name}")):
        gray_batch = sample_batched["image"].cuda()
        img_path = sample_batched.get("file_path", [f"image_{i_batch}"])[0]
        img_name = os.path.basename(img_path).split('.')[0]
        
        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        original_img = gray_batch[0].detach().cpu().numpy().transpose((1, 2, 0))
        
    
        recon_start = time.time()
        gray_rec = model(gray_batch)
        recon_time = time.time() - recon_start
        reconstruction_times.append(recon_time)
        
      
        seg_start = time.time()
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
        seg_time = time.time() - seg_start
        segmentation_times.append(seg_time)
        
   
        total_inference_time = recon_time + seg_time
        total_inference_times.append(total_inference_time)
        status = "anomaly" if is_normal else "normal"
        inference_times_by_status[status].append(total_inference_time)
        
 
        out_mask_averaged = F.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)
        
        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        all_gt_masks.append(true_mask_cv[:, :, 0])
        all_pred_masks.append(out_mask_cv)
        mask_cnt += 1
        

        if visualize:
            vis_start = time.time()
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Anomaly Detection - {img_name} (Score: {image_score:.4f})", fontsize=16)
            axes[0].imshow(original_img); axes[0].set_title("Original"); axes[0].axis('off')
            axes[1].imshow(out_mask_cv, cmap='jet', vmin=0, vmax=1); axes[1].set_title("Heatmap"); axes[1].axis('off')
            axes[2].imshow(original_img); axes[2].imshow(out_mask_cv, cmap='jet', alpha=0.5, vmin=0, vmax=1); 
            axes[2].set_title("Overlay"); axes[2].axis('off')
            plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{img_name}_heatmap.png"), dpi=200); plt.close(fig)
            timing['visualization'] += time.time() - vis_start
    

    timing['inference'] = time.time() - stage_start
    stage_start = time.time()
    

    total_images = len(total_inference_times)
    normal_images = len(inference_times_by_status['normal'])
    anomaly_images = len(inference_times_by_status['anomaly'])
    avg_inference_time = np.mean(total_inference_times)
    median_inference_time = np.median(total_inference_times)
    min_inference_time = np.min(total_inference_times)
    max_inference_time = np.max(total_inference_times)
    std_inference_time = np.std(total_inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    avg_reconstruction_time = np.mean(reconstruction_times)
    avg_segmentation_time = np.mean(segmentation_times)
    avg_normal_time = np.mean(inference_times_by_status['normal']) if inference_times_by_status['normal'] else 0
    avg_anomaly_time = np.mean(inference_times_by_status['anomaly']) if inference_times_by_status['anomaly'] else 0
    
    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt].astype(np.uint8)
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    aupro_score, pro_curve, thresholds = calculate_aupro(all_gt_masks, all_pred_masks)
    timing['metrics'] = time.time() - stage_start
    total_time = time.time() - start_time
    total_time_formatted = str(datetime.timedelta(seconds=int(total_time)))

    results = {
        'auroc': auroc, 'ap': ap, 'auroc_pixel': auroc_pixel, 'ap_pixel': ap_pixel, 'aupro': aupro_score,
        'avg_inference_time': avg_inference_time, 'median_inference_time': median_inference_time,
        'min_inference_time': min_inference_time, 'max_inference_time': max_inference_time,
        'std_inference_time': std_inference_time, 'fps': fps,
        'avg_reconstruction_time': avg_reconstruction_time, 'avg_segmentation_time': avg_segmentation_time,
        'avg_normal_time': avg_normal_time, 'avg_anomaly_time': avg_anomaly_time,
        'total_images': total_images, 'normal_images': normal_images, 'anomaly_images': anomaly_images
    }
    

    print(f"\n----- Results: {obj_name} -----")
    print(f"AUC Image: {results['auroc']:.4f}, AUC Pixel: {results['auroc_pixel']:.4f}, AUPRO: {results['aupro']:.4f}")
    print(f"Avg inference: {results['avg_inference_time']*1000:.2f}ms (± {results['std_inference_time']*1000:.2f}ms)")
    print(f"Range: {results['min_inference_time']*1000:.2f}ms - {results['max_inference_time']*1000:.2f}ms")
    print(f"FPS: {results['fps']:.2f}, Total evaluation time: {total_time_formatted}")
    
    return results

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

    obj_list = ['capsule',
                 'bottle',
                 'carpet',
                 'leather',
                 'pill',
                 'transistor',
                 'tile',
                 'cable',
                 'zipper',
                 'toothbrush',
                 'metal_nut',
                 'hazelnut',
                 'screw',
                 'grid',
                 'wood'
                 ]

    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name)
        