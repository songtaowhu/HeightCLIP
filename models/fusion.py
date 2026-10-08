import torch
from torch import nn
from torch.nn import functional as F

class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock2D, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)
        
        self.W =  nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                     kernel_size=1, stride=1, padding=0)
        

        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        
    def forward(self, x, y):
        
        batch_size = x.size(0)
        
        g_x = self.g(y).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(y).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        yy = torch.matmul(f_div_C, g_x)
        yy = yy.permute(0, 2, 1).contiguous()
        yy = yy.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(yy)
        z = W_y + x
        
        return z
    
if __name__ == '__main__':

    feature1 = torch.randn(1, 512, 8,1)
    feature2 = torch.randn(1, 512, 16,16)
    fusion = NonLocalBlock2D(512)
    feature_fusion = fusion(feature1,feature2)
    print(feature_fusion.shape)