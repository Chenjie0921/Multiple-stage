import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
        #     nn.BatchNorm2d(ch_out),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
        #     nn.BatchNorm2d(ch_out),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            # nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm1d(ch_out),
            # nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		#     nn.BatchNorm2d(ch_out),
		# 	nn.ReLU(inplace=True)
        # )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=1),
            nn.Conv1d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool = nn.MaxPool1d(kernel_size=1, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=1024)
        self.Conv2 = conv_block(ch_in=1024, ch_out=512)
        self.Conv3 = conv_block(ch_in=512, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=128)
        self.Conv5 = conv_block(ch_in=128, ch_out=64)

        self.Up5 = up_conv(ch_in=64, ch_out=128)
        self.Up_conv5 = conv_block(ch_in=128, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=512)
        self.Up_conv3 = conv_block(ch_in=512, ch_out=512)

        self.Up2 = up_conv(ch_in=512, ch_out=1024)
        self.Up_conv2 = conv_block(ch_in=1024, ch_out=2048)

        # self.Conv_1x1 = nn.Conv2d(1024, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)#[32,64,1]-->[32,128,1]
        d5 = torch.add(x4, d5)#[32,128,1]-->[32,128,1]

        d5 = self.Up_conv5(d5)#[32,128,1]-->[32,128,1]

        d4 = self.Up4(d5)#[32,128,1]-->[32,256,1]
        d4 = torch.add(x3, d4)#[32,256,1]-->[32,256,1]
        d4 = self.Up_conv4(d4)#[32,256,1]-->[32,256,1]

        d3 = self.Up3(d4)#[32,256,1]-->[32,512,1]
        d3 = torch.add(x2, d3)#[32,512,1]-->[32,512,1]
        d3 = self.Up_conv3(d3)#[32,512,1]-->[32,512,1]

        d2 = self.Up2(d3)#[32,512,1]-->[32,1024,1]
        d2 = torch.add(x1, d2)#[32,1024,1]-->[32,1024,1]
        d2 = self.Up_conv2(d2)#[32,1024,1]-->[32,2048,1]

        # d1 = self.Conv_1x1(d2)

        return d2
if __name__ == '__main__':
    # unet = U_Net(img_ch=2048,output_ch=2048)
    # rnn = nn.LSTMCell(2048,2048)
    # input = torch.randn(2,32,2048)
    hx = torch.randn(32,2048)
    cx = torch.randn(32,2048)
    inputs = torch.stack((hx,cx),dim=0)
    h0 = torch.randn(2,32,2048)
    # output = []
    # for i in range(input.size()[0]):
    #     hx,cx = rnn(input[i],(hx,cx))
    #     output.append(hx)
    # output = torch.stack(output,dim=0)
    # print(output)
    rnn = nn.RNN(2048,2048,2,bias=True,nonlinearity='relu')
    output,hc = rnn(inputs,h0)
    print(output)
    print(hc)
    # trans = torch.nn.TransformerEncoderLayer(d_model=2048,nhead=8)
    # data = torch.ones([32,2048,1])
    # res = unet(data)
    # summary(trans,(1,2048),batch_size=32,device='cpu')
    # print(unet)
    # conv1 = nn.Conv1d(in_channels=2048,out_channels=1024,kernel_size=3,stride=1,padding=1,bias=True)
    # res = conv1(data)
    # print(res)
    # print(summary(unet,(2,64,2048,1)))