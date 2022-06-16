import oneflow as flow
import oneflow.nn as nn
import flowvision.transforms as transforms
import numpy as np
import string
from PIL import  Image
characters = '-' + string.digits + string.ascii_letters
width, height, n_len, n_classes = 192, 64, 4, len(characters)
n_input_length = 12

class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 192)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        pools = [2, 2, 2, 2, (2, 1)]

        def block(num_convs, in_channels, out_channels):
            lys = []
            for _ in range(num_convs):
                lys.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(1, 1)))
                lys.append(nn.BatchNorm2d(out_channels))
                lys.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            return nn.Sequential(*lys)

        convs_blks = []
        last_channels = 3
        for  (num_convs, out_channels, k_pool) in zip( layers, channels, pools):
            convs_blks.append(block(num_convs, last_channels,out_channels))
            last_channels = out_channels
            convs_blks.append(nn.MaxPool2d(k_pool))
        
        self.cnn = nn.Sequential(*convs_blks,nn.Dropout(0.25, inplace=True),)
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_classes)
    
    def infer_features(self):
        x = flow.zeros((1,)+self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class Recognize():
    def __init__(self):
        self.model = Model(n_classes, input_shape=(3, height, width)) #.cuda()
        self.params = flow.load("./model")
        self.model.load_state_dict(self.params)
        self.model.eval()

    def rm_dup(self,s):
        for i in range(len(s)-1):
            if (s[i] == s[i+1] and s[i]!='-'):
                s = np.delete(s,i)
                return s
        return s[s!='-'][:4]


    def decode(self,sequence):
        s = np.array([characters[x]  for x in sequence])
        while((s !='-').sum()>4):
            s = self.rm_dup(s)
        s =s[s!='-']
        if len(s) == 0:
            return ''
        return ''.join(s)

    def recognize(self,im):
        im = im.resize((192, 64), Image.ANTIALIAS)   
        transf = transforms.ToTensor()
        img_tensor = transf(im)  # 模型输入为0-1
        output = self.model(img_tensor.unsqueeze(0)) #.cuda()
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        return self.decode(output_argmax[0])