import torch
import torch.nn as nn
import torch.nn.functional as F

'''Simple classifier for SA'''

class EasyClassifier(nn.Module):
    def __init__(self, img_size=28**2, h_dim=200):
        super(EasyClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(img_size, h_dim),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.layer3 = nn.Linear(h_dim, 10)
    
    def up2lyr1(self, x):
        return self.layer1(x)
    
    def up2lyr2(self, x):
        return self.layer2(self.layer1(x))
    
    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

class MnistClassifier(nn.Module):
    '''Same architecture as in original SA paper'''
    def __init__(self, img_size=32):
        super(MnistClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(3136, 512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 10), # regularization only on linear
            ),
        )
    
    def at_by_layer(self, x, layer_idx = 0):
        current_idx = 0
        out = x
        for lyr in self.conv_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                at = torch.mean(out, dim=3)
                at = torch.mean(at, dim=2)
                return at
            else:
                current_idx += 1
        out = out.view(-1, 64*7*7)
        if current_idx == layer_idx:
            return out
        else:
            current_idx += 1
        for lyr in self.dense_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                return out
            else:
                current_idx += 1
        # should never reach here
        raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 64*7*7)
        out = self.dense_layers(out)
        return out
    
class CifarClassifier(nn.Module):
    '''Same architecture as in original SA paper'''
    def __init__(self, img_size=32):
        super(CifarClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(2048, 1024),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(1024, 512), # regularization only on linear
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 10), # regularization only on linear
            ),
        )
    
    def at_by_layer(self, x, layer_idx = 0):
        current_idx = 0
        out = x
        for lyr in self.conv_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                at = torch.mean(out, dim=3)
                at = torch.mean(at, dim=2)
                return at
            else:
                current_idx += 1
        out = out.view(-1, 128*4*4)
        if current_idx == layer_idx:
            return out
        else:
            current_idx += 1
        for lyr in self.dense_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                return out
            else:
                current_idx += 1
        # should never reach here
        raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 128*4*4)
        out = self.dense_layers(out)
        return out
    
class ConvBN_mobilenet(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvBN_mobilenet, self).__init__()
        self.conv_lyr = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv_lyr(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ConvDW_mobilenet(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvDW_mobilenet, self).__init__()
        self.conv_lyr1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        # nn.ReLU(inplace=True),

        self.conv_lyr2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        # nn.ReLU(inplace=True),
    
    def forward(self, x):
        out = self.conv_lyr1(x)
        out = F.relu(self.bn1(out))
        out = self.conv_lyr2(out)
        out = F.relu(self.bn2(out))
        return out
    
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        
        self.layerM_01 = ConvBN_mobilenet(3, 32, 2) # if MNIST import fails, probably just because of channels
        self.layerM_02 = ConvDW_mobilenet(32, 64, 1)
        self.layerM_03 = ConvDW_mobilenet(64, 128, 1)
        self.layerM_04 = ConvDW_mobilenet(128, 128, 1)
        self.layerM_05 = ConvDW_mobilenet(128, 256, 2)
        self.layerM_06 = ConvDW_mobilenet(256, 256, 1)
        self.layerM_07 = ConvDW_mobilenet(256, 512, 2)
        self.layerM_08 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_09 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_10 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_11 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_12 = ConvDW_mobilenet(512, 512, 1)
        self.layerM_13 = ConvDW_mobilenet(512, 1024, 2)
        self.layerM_14 = ConvDW_mobilenet(1024, 1024, 1)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.layerM_01(x)
        out = self.layerM_02(out)
        out = self.layerM_03(out)
        out = self.layerM_04(out)
        out = self.layerM_05(out)
        out = self.layerM_06(out)
        out = self.layerM_07(out)
        out = self.layerM_08(out)
        out = self.layerM_09(out)
        out = self.layerM_10(out)
        out = self.layerM_11(out)
        out = self.layerM_12(out)
        out = self.layerM_13(out)
        out = self.layerM_14(out)
        out = self.avgpool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out

class VGGBlock(nn.Module):
    def __init__(self, inp, outp, lnum):
        super(VGGBlock, self).__init__()
        self.first_lyr = nn.Sequential(
            nn.Conv2d(inp, outp, kernel_size=3, padding=1), 
            nn.BatchNorm2d(outp), 
            nn.ReLU(inplace=True)
        )
        leftover_lyrs = []
        for li in range(1, lnum):
            leftover_lyrs.append(nn.Conv2d(outp, outp, kernel_size=3, padding=1))
            leftover_lyrs.append(nn.BatchNorm2d(outp)) # uses BN as default
            leftover_lyrs.append(nn.ReLU(inplace=True))
        self.next_lyrs = nn.Sequential(*leftover_lyrs)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.first_lyr(x)
        out = self.next_lyrs(out)
        out = self.maxpool(out)
        return out
    
class VGGClassifier(nn.Module):
    def __init__(self, inp, class_num):
        super(VGGClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(inp, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.layer1 = VGGBlock(3, 64, 2)
        self.layer2 = VGGBlock(64, 128, 2)
        self.layer3 = VGGBlock(128, 256, 4)
        self.layer4 = VGGBlock(256, 512, 4)
        self.layer5 = VGGBlock(512, 512, 4)
        self.vggc = VGGClassifier(512 * 1 * 1, 10) # 10 should be immutable...
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.vggc(out) # returns 2d
        return out