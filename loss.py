import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

epsilon = torch.tensor(1e-8)

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            
            input = input[mask]
            target = target[mask]
            
        g = torch.log(input) - torch.log(target)  
            
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85*(norm**2)) * (torch.sum(g))**2
        
        Dg = torch.var(g, unbiased = True) + 0.15 * torch.pow(torch.mean(g), 2)
        
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

class ScoreLoss(nn.Module):  # score map loss
    def __init__(self):
        super().__init__()
        self.name = "ScoreLoss"
        
    def forward(self, score_map, target, edges, upscale_factor):

        original_size = score_map.shape[2:]  # width and height
        new_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
        upsampled_score_map = F.interpolate(score_map, size=new_size, mode='bilinear', align_corners=False)
        
        edges = torch.as_tensor(edges, dtype=upsampled_score_map.dtype, device=upsampled_score_map.device)
        
        discretized = torch.bucketize(target, edges, right=False) - 1 
        
        discretized = torch.clamp(discretized, 0, len(edges)-1)
                   
        loss_fn = torch.nn.CrossEntropyLoss()  
        discretized = discretized.squeeze(dim=1) 
        
        upsampled_score_map = upsampled_score_map / 0.07  # Temperature scaling 
        loss = loss_fn(upsampled_score_map, discretized)
        return loss