import torch
import torch.nn as nn
import utils.brucenet as brucenet
import spyramid as sp
import utils.argmax_diff as amxd
import torch.nn.functional as F

class StatNetEncoder(nn.Module):
    def __init__(self, img_size, batch_size, num_stats, device, entropy=False):
        super(StatNetEncoder, self).__init__()
        
        #self.onehot = onehot #use a hard onehot constraint
        self.entropy = entropy #use entropy instead of L1 weight loss
        #image params
        self.insize_y, self.insize_x = img_size        
        self.batch_size = batch_size
        self.num_stats = num_stats
        self.device = device
        self.color_channels = 1 #color channels
        self.dummy_img = torch.zeros(self.batch_size,
                                     self.color_channels,
                                     self.insize_y,
                                     self.insize_x).to(device)
        
        #bruceNet params (single pooling region)
        self.poolingsize = 1e20 #1e20 for single pooling region# 128 corresponds to 10 degrees eccentricity on eyelink
        self.bruceNet_nstats = 150 #number of stats output from brucenet for single pooling region
        self.bottleneck_size = self.num_stats #number of stats we'll compress down to
        #one columns and rows to penalize mulitple stats per output and multiple outputs per stat.
        self.one_columns = torch.ones(self.num_stats, dtype=torch.float32,requires_grad=True).to(device)
        self.one_rows = torch.ones(self.bruceNet_nstats,dtype=torch.float32,requires_grad=True).to(device)
        
        # Added by CK
        #self.batch_norm = nn.BatchNorm1d(self.bruceNet_nstats)
        self.batch_norm = nn.BatchNorm1d(50)
        self.layer_norm = nn.LayerNorm(150) # Normalizing the input statistics
        
        #BruceNet Encoder Layer
        self.bruceNet = brucenet.BruceNet(pooling_region_size=self.poolingsize,
                                          pyramid_params=None,
                                          dummy_img=self.dummy_img).cuda(self.device)
        #encoder, calls Brucenet forward pass (which calculates statistics)
        self.encoder = nn.Sequential(
            self.bruceNet
        )
        #self.w = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty((self.bruceNet_nstats, self.bottleneck_size), device=self.device))) #added this line of code von CK

        self._w = nn.Parameter(torch.rand((self.bruceNet_nstats, self.bottleneck_size), device=self.device))
        #self.w = torch.nn.functional.softmax(self._w,dim=0)
        #self.w = nn.Parameter(torch.divide(torch.ones((self.bruceNet_nstats,self.bottleneck_size), device=self.device, requires_grad=True),2.))
        #self.wa = torch.nn.functional.softmax(self.w)

        def compressor(x):
            #self.w = torch.sigmoid(self._w) #habe ich ausgemacht
            self.w = torch.nn.functional.softmax(self._w,dim=0)
            x = torch.matmul(x,self.w)
            #x = torch.relu(x)  # Adding ReLU activation for nonlinearity
            return(x)
               
        self.compressor = compressor

    
    def sparse_loss(self):
        '''loss on sparsity of columns'''
        #sparsity = torch.sum(torch.abs(self.w)) # mein versuch
        
        #return(torch.mean(self.w.norm(dim=0,p=1)))
        sparsity = (torch.norm(self.w,dim=0,p=1) - self.one_columns).norm(p=1) #das war das letzte auskommentierte
        #sparsity = (torch.nn.functional.gumbel_softmax(self.w).norm(p=0,dim=0) - self.one_columns).norm(p=1)
        #sparsity = torch.sum(self.w)
        #sparsity = torch.mean((self.w).norm(dim=0,p=0))
        #sparsity = torch.sum(sparsity - self.one_columns)
        #print(sparsity)
        #print(sparsity_loss)
        #sparsity_loss = torch.sum(torch.abs(torch.norm(self.w,axis=0) - self.one_columns))
        return(sparsity)
    
    def entropy_loss(self):
        p = torch.nn.functional.log_softmax(self._w,dim=0)
        p = torch.clamp_min(p, .0000001)
        ent = -(p * torch.log(p)).sum(dim=0)
        ent = ent.sum()
        #ent = torch.exp(ent).sum() #undo log and sum to scalar output
        return(ent)
        
    def norm_weights(self):
        #print(self.w.ma
        self.w.data = torch.clamp(self.w.data,0,1)
        #self.w.data = torch.div(self.w.data, self.w.data.max(dim=0))
        return()
    
    def multistat_loss(self):
        '''penalize sum of columsn to be less than 1'''
        diff = self.one_rows - self.w.sum(dim=1)
        diff = diff.clamp_(0,1).sum()
        return(diff)
    
    def getsstatlabels(self, device, tempstats=True):
        dummy_img = self.dummy_img.to(device)
        stats_labels = self.bruceNet.calcstats(dummy_img, get_labels=True)
        return(stats_labels)
    
    def forward(self, x):
        #print('input',x.shape) #input torch.Size([200, 1, 64, 64])

        x = self.encoder(x)
                # Normalize the statistics
        #x = self.layer_norm(x)  # Normalizing the input statistics
        #x = self.batch_norm(x)  # Apply batch normalization
        #mean = torch.mean(x, dim=0, keepdim=True)
        #std = torch.std(x, dim=0, keepdim=True) + 1e-7  # Adding a small value to prevent division by zero
        #x = (x - mean) / std
        x = self.compressor(x)
        #x = F.normalize(x, p=2, dim=1) #added CK L2 normalization, might help might not
        #print('compressed',x.shape) #compressed torch.Size([200, 50])
        return(x)
