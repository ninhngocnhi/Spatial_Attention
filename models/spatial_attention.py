import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

GO = 0
EOS_TOKEN = 1             

from torchvision.models.inception import BasicConv2d, InceptionA

class OneHot(nn.Module):
    def __init__(self, depth):
        super(OneHot, self).__init__()
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        emb.weight.requires_grad = False
        self.emb = emb
        self.depth = depth

    def forward(self, input):
        return self.emb(input)

class Incept(nn.Module):
    def __init__(self):
        super(Incept, self).__init__()
        self.conv_1 = BasicConv2d(3, 32, kernel_size = 3, stride = 2)
        self.conv_2 = BasicConv2d(32, 32, kernel_size = 3)
        self.conv_3 = BasicConv2d(32, 64, kernel_size = 3)
        self.pool = torch.nn.MaxPool2d(3, stride=2)
        self.conv_4 = BasicConv2d(64, 80, kernel_size = 1)
        self.conv_5 = BasicConv2d(80, 192, kernel_size = 3)
        self.mix_1 = InceptionA(192, pool_features=32)
        self.mix_2 = InceptionA(256, pool_features=64)
        self.mix_3 = InceptionA(288, pool_features=64)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.pool(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.pool(x)
        x = self.mix_1(x)
        x = self.mix_2(x)
        x = self.mix_3(x)
        # print("the out of my incept: ", x.shape) #[32,288,34,34]
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_dec, feature_map ):
        timestep = feature_map.size(1)
        # print(feature_map.size())
        hidden_dec = hidden_dec.expand(timestep, -1, -1).transpose(0,1)
        # print(hidden_dec.size())
        score = torch.tanh(self.attn(torch.cat([hidden_dec, feature_map], 2)))
        atten_weight = score.softmax(dim = 2)
        context = torch.sum(atten_weight * feature_map, dim = 1)
        return context, atten_weight

class Decoder(nn.Module):
    def __init__(self, nEm, nHid):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(nEm, nHid)
        self.atten = Attention(nHid)
        self.linear = nn.Linear(nHid*2, nHid)
        self.rnn = nn.GRU(nHid, nHid, batch_first=False)
        self.out = nn.Linear(nHid, nEm)
        self.outt = nn.Softmax()
    
    def forward(self, targets, hidden_dec, features):
        emb_in = self.emb(targets)
        context, atten_weight = self.atten(hidden_dec, features)
        # print("shape context, atten_weight", context.size(), atten_weight.size()) #[32,256], [32,1156,256]
        # print("shape target ", emb_in.size()) #[32,256]
        # print("shape concat:
        #  ", torch.cat([emb_in, context],1).size())
        in_rnn = self.linear(torch.cat([emb_in, context],1)).unsqueeze(0)
        # print("shape in_rnn: ",in_rnn.size()) #[1,32,256]
        # print("shape hidden_dec: ", hidden_dec.size()) #[1,32,256]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        output, current_hid = self.rnn(in_rnn, hidden_dec)
        # print("shape output, current_hid", output.size(), current_hid.size()) #[1,32,256], [1,32,256]
        out_pre = self.linear(torch.cat([output.squeeze(0),context], 1))
        # print(out_pre.size())
        out_pre = self.outt(self.out(out_pre))
        return out_pre, current_hid

class Model(nn.Module): 
    def __init__(self, nHid, nEm, imgH, imgW):
        super().__init__()
        self.incept = Incept()
        self.decoder = Decoder(nEm, nHid)
        self.nHid = nHid
        f = self.incept(torch.rand(1, 3, imgH, imgW))
        self._fh = f.size(2)
        self._fw = f.size(3)
        self.onehot_x = OneHot(self._fh)
        self.onehot_y = OneHot(self._fw)
        self.li = nn.Linear(288 + self._fh + self._fw, nHid)
        self.imgH = imgH
        self.imgW = imgW
        
    def forward(self, targets, hid_dec, images):
        device = images.device
        feature_map = self.incept(images)
        b, c, h, w = feature_map.size()
        x, y = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
        h_loc = self.onehot_x(x)
        w_loc = self.onehot_y(y)
        loc = torch.cat([h_loc, w_loc], dim = 2).unsqueeze(0).expand(b, -1, -1, -1)
        features = torch.cat([feature_map.permute(0, 2, 3, 1), loc], dim=3)
        features = features.contiguous().view(b, -1, 288 + self._fh + self._fw)
        features = self.li(features)
        out_pre, current_hid = self.decoder(targets, hid_dec, features)
        return out_pre, current_hid

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.nHid))
        return result