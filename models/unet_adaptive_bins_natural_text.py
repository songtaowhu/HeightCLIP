import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import time
from .fusion import NonLocalBlock2D
from .miniViT import mViT
# from miniViT import mViT

from models.clip.clip_base import *

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__() 

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())
    
    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1) 
        return self._net(f) 


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features) 

        # self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2048 + 8, 2048, kernel_size=1, stride=1, padding=1)  # 8 is channel of score map
        
        self.up1 = UpSampleBN(skip_input=features + 112 + 64, output_features=features // 2) 
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4) 
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8) 
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features): 
  
        x_block0, x_block1, x_block2, x_block3, x_block4, x_block5 = features[4], features[5], features[6], features[8], features[11], features[-1]  # last is score map feature
        
        x_block4 = torch.cat((x_block4, x_block5),dim = 1)
        x_d0 = self.conv2(x_block4) 

        x_d1 = self.up1(x_d0, x_block3) 
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1).contiguous()
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

        cls_pos = self.positional_embedding[0:1, :].to(x.dtype)
        spatial_pos = F.interpolate(
            self.positional_embedding[1:, ].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3,1,2).contiguous(),
            size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H * W).permute(1, 0).contiguous().to(x.dtype)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)

        x = x + positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        x = x.permute(1, 2, 0).contiguous()
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map


class Encoder(nn.Module):
    def __init__(self, backend): 
        super(Encoder, self).__init__()
        self.original_model = backend
        embed_dim = 64 * 32
        input_resolution = 512
        heads = 32
        output_dim = 512
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items(): 
            if (k == 'blocks'): 
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1])) 
                
        temp = features[11] 
        x_global, x_local = self.attnpool(temp)
        features.append([x_global, x_local])        
        
        return features 

class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, clip_model, gpu, n_bins=100, min_val=0, max_val=25.5, norm='linear'):
        super(UnetAdaptiveBins, self).__init__()
        
        self.clip_model = clip_model
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.gpu=gpu
        
        self.context_length = 8
        self.token_embed_dim = 512
        self.range_num = 8
        
        self.contexts = nn.Parameter(torch.randn(1, self.context_length, self.token_embed_dim))

        nn.init.trunc_normal_(self.contexts)

        # self.context_decoder = ContextDecoder(visual_dim=512)

        self.gamma = nn.Parameter(torch.ones(512) * 1e-4)
        self.gamma2 = nn.Parameter(torch.ones(512) * 1e-4) 
        
        self.encoder = Encoder(backend) 
        
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)
        
        
        self.decoder = DecoderBN(num_classes=128)
        # self.decoder = DecoderBN(num_features=2560, num_classes=128, bottleneck_features=2560)
        
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        
        self.fusion = NonLocalBlock2D(512)
    
    
    def forward(self, x, **kwargs): 
    
        encoded_features = self.encoder(x)
        global_f, visual_embeddings = encoded_features[-1]
        B, C, H, W = visual_embeddings.shape
        
        bin_edges_subset = torch.linspace(0,25.5,self.range_num + 1)
        
        bin_edges_subset = torch.round(bin_edges_subset)
        
        CaBins = [ f"the object in a remote sensing image is between {int(bin_edges_subset[i])} and {int(bin_edges_subset[i+1])} meters in height"
    for i in range(len(bin_edges_subset) - 1)]
             
        texts = torch.cat([tokenize(c, context_length=22) for c in CaBins]).to(self.gpu)  ## [range_num,30]
        text_bs = texts.unsqueeze(0).expand(B,-1,-1)  # [bs,range_num,30]

        learnable = self.contexts.to(self.gpu) # [1,8,512]
        learnable = learnable.expand(B, -1, -1) # [8,8,512]
        text_features = self.clip_model.encode_learnable_text(text_bs, learnable)
        
        text_features = text_features.permute(0,2,1).unsqueeze(-1)  # (bs,8,512) ->(bs,512,8)->(bs, 512, 8, 1)     
        
        ## reciproal cross-attention fusion
        visual_embeddings = self.fusion(visual_embeddings,text_features) # （bs, 512, 16, 16）       
        text_features = self.fusion(text_features,visual_embeddings).squeeze(-1).permute(0,2,1)  # (bs, 512, 8, 1) -> (bs, 512,8) ->(bs,8,512)
        
        visual = F.normalize(visual_embeddings, dim=1, p=2) # (8,512,16,16)
        text = F.normalize(text_features, dim=2, p=2) # (8,8,512)
         
        score_map = torch.einsum('bchw,bkc->bkhw', visual, text) # (8,8,16,16)
        encoded_features.append(score_map)
    
        
        unet_out = self.decoder(encoded_features, **kwargs)
               
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out) 
        out = self.conv_out(range_attention_maps) 

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        
        return bin_edges, pred, score_map 


    # def get_1x_lr_params(self):  # lr/10 learning rate
    #     return self.encoder.parameters()
  
    # def get_10x_lr_params(self):  # lr learning rate
    #     modules = [self.decoder, self.adaptive_bins_layer, self.conv_out, self.context_decoder]
    #     for m in modules:
    #         yield from m.parameters()
            
    @classmethod
    def build(cls, clip_model, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'
        print('Loading base model ()...'.format(basemodel_name), end='')
        
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        
        m = cls(basemodel, clip_model, n_bins=n_bins, **kwargs) 
        print('Done.')
        return m
    

if __name__ == '__main__':
    st1 = time.time()
    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 512, 512) 
    bins, pred, score = model(x)
    print(bins.shape, pred.shape, score.shape)
