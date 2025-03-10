import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SDConv_direct(nn.Module):
    def __init__(self, input_chanel, output_chanel, lambda_):
        super(SDConv_direct,self).__init__()
        self.inch = input_chanel
        self.outch = output_chanel
        self.lambda_ = lambda_
        self.padding = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.weight = nn.Parameter(torch.empty(output_chanel, input_chanel, 3, 3))
        nn.init.uniform_(self.weight, -0.1, 0.1)
    
    def forward(self, U, V):
        assert U.size() ==  V.size()
        bs = U.size(0)
        out = torch.empty(bs, self.outch, U.size(2), U.size(3)).cuda()
        U = self.padding(U)
        V = self.padding(V)
        for i in range(out.size(2)):
            for j in range(out.size(3)):
                for k in range(self.outch):
                    for n in range(bs):
                        u1_u2 = U[n,:,i:i+3,j:j+3] - U[n,:,i+1,j+1][:,None,None]
                        v1_v2 = V[n,:,i:i+3,j:j+3] - V[n,:,i+1,j+1][:,None,None]
                        S = 1/torch.sqrt(1+ (torch.square(torch.abs(v1_v2))/(self.lambda_)**2))
                        out[n,k,i,j] = torch.sum(self.weight[k] * S * u1_u2)
        return out

class SDConv(nn.Module):
    '''
    paper: NeurIPS 2022 'Semantic Diffusion Network for Semantic Segmentation'
    
    Fixed parameter : kernel size = (3, 3)
                    : padding = 1
                    : stride = 1
    Variable parameter : input_chanel
                        : output_chanel
                        : lambda_
    input : U = Feature activation
            V = Semantic guidance graph
    output : boundary-enhanced feature map
        U, V, and output all have the same shape
    '''
    def __init__(self, input_chanel, output_chanel, lambda_) :
        super(SDConv, self).__init__()
        self.inch = input_chanel
        self.outch = output_chanel
        self.lambda_ = lambda_
        self.weight = nn.Parameter(torch.empty(output_chanel, input_chanel, 3, 3))
        nn.init.uniform_(self.weight, -0.1, 0.1)
        self.unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)

    def forward(self, U, V):
        assert U.size() ==  V.size()
        (BN, C, W, H) = U.size()

        U_unfold = self.unfold(U).view((BN, C, -1, W, H))
        U_unfold_cen = U_unfold[:,:, 4,:,:][:,:,None,:,:]
        dif_U_unfold = U_unfold - U_unfold_cen

        V_unfold = self.unfold(V).view((BN, C, -1, W, H))
        V_unfold_cen = V_unfold[:,:, 4,:,:][:,:,None,:,:]
        dif_V_unfold = V_unfold - V_unfold_cen
        S_V_unfold = 1/torch.sqrt(1+((torch.square(torch.abs(dif_V_unfold)))/self.lambda_**2))

        S_U = torch.mul(dif_U_unfold, S_V_unfold)

        output = torch.einsum("ijklm,njk->inlm", S_U, self.weight.view((self.outch, self.inch, 3*3)))
        return output

class SDNet(nn.Module):
    def __init__(self, input_chanel, output_chanel, lambda_, SDConv = SDConv) :
        super(SDNet,self).__init__()
        self.SDC = SDConv(input_chanel, output_chanel, lambda_)
        self.BN = nn.BatchNorm2d(output_chanel)
        self.relu = nn.ReLU()
        self.out = nn.Conv2d((input_chanel+output_chanel), input_chanel, 1)

    def forward(self, U, V):
        x = self.SDC(U,V)
        x = self.relu(self.BN(x))
        x = torch.cat((U, x), dim=1)
        out = self.out(x)
        assert out.size() == U.size()
        return out

#data1 = torch.randn(5,4,500,500).cuda()
#data2 = torch.randn(5,4,500,500).cuda()
#weight = torch.randn(5,4,3,3).cuda()

#SDN = SDConv(4,5,1).cuda()
#SDN2 = SDConv_direct(4,5,1).cuda()
#import time
#t1 = time.time()
#a1 = SDN(data1, data2)
#t2 = time.time()

#t3 = time.time()
#out = F.conv2d(data1, weight)
#t4 = time.time()

#print(t2-t1,t3-t2,t4-t3)

