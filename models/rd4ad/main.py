
import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from rd4ad_test import evaluation, visualization, test
from torch.nn import functional as F
from tqdm.notebook import tqdm


# adjusted logic for paperclips dataset, gc colection and memory management are explicitly added, amp training enabled, gradscaler for amp, LR scheduler added, saves best and latest model based on auroc
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(_class_):
    print(f"Training model for {_class_}")
    epochs = 200
    learning_rate = 0.005
    batch_size = 8
    image_size = 256

    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    if _class_ == 'paperclips_front':
        train_path = r'C:/Thesis/Data/paperclips_sorted/paperclips_front/train'
        test_path = r'C:/Thesis/Data/paperclips_sorted/paperclips_front'
    elif _class_ == 'paperclips_side':
        train_path = r'C:/Thesis/Data/paperclips_sorted/paperclips_side/train'
        test_path = r'C:/Thesis/Data/paperclips_sorted/paperclips_side'
    else:
        train_path = './mvtec/' + _class_ + '/train'
        test_path = './mvtec/' + _class_

    os.makedirs('./checkpoints', exist_ok=True)
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    
  
    num_workers = 4  
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=1, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"Training dataset size: {len(train_data)}")
    print(f"Testing dataset size: {len(test_data)}")


    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval() 
    
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    best_auroc_sp = 0
    best_epoch = 0

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for img, label in pbar:
            img = img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # More memory efficient than .zero_grad()
            
            try:
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():  
                            inputs = encoder(img)
                        outputs = decoder(bn(inputs))
                        loss = loss_fucntion(inputs, outputs)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with torch.no_grad(): 
                        inputs = encoder(img)
                    outputs = decoder(bn(inputs))
                    loss = loss_fucntion(inputs, outputs)
                    loss.backward()
                    optimizer.step()
                current_loss = loss.item()
                loss_list.append(current_loss)
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                
            except Exception as e:
                print(f"Error during training: {e}")
            
            finally:
                del img
                del inputs
                del outputs
                del loss
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
        epoch_loss = np.mean(loss_list)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')
        
        scheduler.step(epoch_loss)
 
        if device == 'cuda':
            torch.cuda.synchronize() 
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        if (epoch + 1) % 10 == 0:
            print(f"Current CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print("Evaluating model...")
            bn.eval()
            decoder.eval()
            
            with torch.no_grad(): 
                auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            
            print(f'Pixel AUROC: {auroc_px:.3f}, Sample AUROC: {auroc_sp:.3f}, Pixel AUPRO: {aupro_px:.3f}')

            bn.train()
            decoder.train()

            torch.save({
                'epoch': epoch + 1,
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'auroc_px': auroc_px,
                'auroc_sp': auroc_sp,
                'aupro_px': aupro_px
            }, ckp_path)
            
            if auroc_sp > best_auroc_sp:
                best_auroc_sp = auroc_sp
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch + 1,
                    'bn': bn.state_dict(),
                    'decoder': decoder.state_dict(),
                    'auroc_px': auroc_px,
                    'auroc_sp': auroc_sp,
                    'aupro_px': aupro_px
                }, ckp_path.replace('.pth', '_best.pth'))
                print(f"New best model saved at epoch {epoch+1}")

            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    print(f"Training completed. Best Sample AUROC: {best_auroc_sp:.4f} at epoch {best_epoch}")
 
    checkpoint = torch.load(ckp_path.replace('.pth', '_best.pth'))
    bn.load_state_dict(checkpoint['bn'])
    decoder.load_state_dict(checkpoint['decoder'])

    print("Performing final evaluation with best model...")
    bn.eval()
    decoder.eval()
    with torch.no_grad():
        final_auroc_px, final_auroc_sp, final_aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
    print(f'Final results - Pixel AUROC: {final_auroc_px:.4f}, Sample AUROC: {final_auroc_sp:.4f}, Pixel AUPRO: {final_aupro_px:.4f}')
    
    return final_auroc_px, final_auroc_sp, final_aupro_px



if __name__ == '__main__':

    setup_seed(111)
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        train(i)