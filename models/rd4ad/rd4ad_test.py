import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import time 
from sklearn.metrics import average_precision_score
import numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import auc
import os


# added aupro calculation, timing statistics, refined save vis function



def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def evaluation(encoder, bn, decoder, dataloader, device, _class_=None):
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    inference_times = []
    
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            start_time = time.time()

            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            if device == 'cuda':
                torch.cuda.synchronize()
  
            end_time = time.time()

            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            if label.item() != 0:
                gt_np = gt.squeeze(0).cpu().numpy().astype(int)
                if gt_np.ndim > 2:  
                    gt_binary = np.max(gt_np, axis=0)  
                else:
                    gt_binary = gt_np 
                gt_binary = gt_binary[np.newaxis, :, :] 
                anomaly_map_expanded = anomaly_map[np.newaxis, :, :] 
                assert gt_binary.ndim == 3, f"gt_binary.ndim should be 3, got {gt_binary.ndim}"
                assert anomaly_map_expanded.ndim == 3, f"anomaly_map_expanded.ndim should be 3, got {anomaly_map_expanded.ndim}"
                assert gt_binary.shape == anomaly_map_expanded.shape, f"Shape mismatch: gt_binary {gt_binary.shape} vs anomaly_map_expanded {anomaly_map_expanded.shape}"
                
                aupro_list.append(compute_aupro(gt_binary, anomaly_map_expanded))
            
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())

            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)

        aupro_px = round(np.mean(aupro_list), 3) if aupro_list else 0.0
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 3)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 3)

    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    fps = 1000 / avg_time  

    print(f"\nTiming Statistics:")
    print(f"  Average Inference Time: {avg_time:.2f} ms")
    print(f"  Standard Deviation: {std_time:.2f} ms")
    print(f"  Minimum Inference Time: {min_time:.2f} ms")
    print(f"  Maximum Inference Time: {max_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")

    timing_stats = {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps,
        'all_times_ms': inference_times
    }
    
    return auroc_px, auroc_sp, aupro_px, ap_px, ap_sp, timing_stats

def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = r"C:\Thesis\Data\paperclips_sorted\\" + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    auroc_px, auroc_sp, aupro_px, ap_px, ap_sp, timing_stats = evaluation(encoder, bn, decoder, test_dataloader, device, _class_)

    print(f"{_class_}: AUROC-PX={auroc_px}, AUROC-SP={auroc_sp}, AUPRO-PX={aupro_px}, AP-PX={ap_px}, AP-SP={ap_sp}")
    

    return {
        'auroc_px': auroc_px,
        'auroc_sp': auroc_sp,
        'aupro_px': aupro_px,
        'ap_px': ap_px,
        'ap_sp': ap_sp
    }


def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = r"C:\Thesis\Data\paperclips_sorted\\" + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0):
                continue
                
            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            anomaly_score = np.mean(anomaly_map)
            
            ano_map = min_max_norm(anomaly_map)
            ano_map = 1 - ano_map
            ano_map = cvt2heatmap(ano_map*255)
            
            img_np = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img_np = np.uint8(min_max_norm(img_np)*255)

            overlay = show_cam_on_image(img_np, ano_map)

            save_dir = r"C:\Thesis\Results\rd4ad"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(ano_map)
            plt.title(f'Heatmap (Score: {anomaly_score:.4f})')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.suptitle(f'Sample {count} - Anomaly Score: {anomaly_score:.4f}')
            plt.tight_layout()

            combined_filename = os.path.join(save_dir, f'sample_{count}_score_{anomaly_score:.4f}.png')
            plt.savefig(combined_filename, bbox_inches='tight', dpi=150)
            plt.close() 
            
            count += 1


def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            #if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            #print(inputs[-1].shape)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig('ad.png')
            #plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            #count += 1
            #if count>40:
            #    return 0
                #assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        #print(type(vis_data))
        #np.save('vis.npy',vis_data)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


def compute_aupro(masks, amaps, num_thresholds=100, max_fpr=0.3):

    if isinstance(masks, list):
        masks = np.stack(masks)
    if isinstance(amaps, list):
        amaps = np.stack(amaps)
    
    assert amaps.shape == masks.shape, "amaps and masks must have same shape"
    
    min_th = amaps.min()
    max_th = amaps.max()
    thresholds = np.linspace(min_th, max_th, num_thresholds)

    fprs = np.zeros(num_thresholds)
    pros = np.zeros(num_thresholds)

    for i, threshold in enumerate(thresholds):
        binary_preds = (amaps > threshold).astype(np.uint8)
        
        pro_sum = fp_pixels = 0
        region_count = normal_pixels = 0

        for mask, binary_pred in zip(masks, binary_preds):
            labeled_mask, num_regions = ndi.label(mask)
            
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_mask == region_id)
                region_size = np.sum(region_mask)

                region_pred_overlap = np.logical_and(binary_pred, region_mask).sum()
                pro = region_pred_overlap / region_size
                pro_sum += pro
                region_count += 1

            inverse_mask = (mask == 0)
            fp_pixels += np.logical_and(inverse_mask, binary_pred).sum()
            normal_pixels += inverse_mask.sum()

        pros[i] = pro_sum / region_count if region_count > 0 else 0
        fprs[i] = fp_pixels / normal_pixels if normal_pixels > 0 else 0

    if region_count == 0:
        return np.nan

    valid_indices = fprs <= max_fpr
    if not np.any(valid_indices):
        return np.nan
    
    filtered_fprs = fprs[valid_indices]
    filtered_pros = pros[valid_indices]

    sort_idx = np.argsort(filtered_fprs)
    sorted_fprs = filtered_fprs[sort_idx]
    sorted_pros = filtered_pros[sort_idx]

    if sorted_fprs.max() > 0:
        norm_fprs = sorted_fprs / sorted_fprs.max()
    else:
        norm_fprs = sorted_fprs

    return auc(norm_fprs, sorted_pros)

def detection(encoder, bn, decoder, dataloader,device,_class_):
    #_, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)


            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1


        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean