import logging
import math
import torch.nn as nn
import torch

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )


def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = None
        self.bn1 = None
        self.relu1 = None
        self.conv2 = None
        self.bn2 = None
        self.relu2 = None
        ################################################################################
        # TODO:                                                                        #
        #  define the network                                                          #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):
        out = None
        ################################################################################
        # TODO:                                                                        #
        #  finish the forward function                                                 #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, gamma_zero = False, avg_pool_downsample = False):
        self.inplanes = 64
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock ):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        ################################################################################
        # TODO: build the network                                                      #
        # block: the block class                                                       #
        # planes: number of channels within each block                                 #
        #         (may be different from the number of input channels)                 #
        # blocks: number of blocks in the layer                                        #
        # stride: the stride of the first block, stride > 1 means downsampling.        #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
        	# 降采样通过conv 实现
            downsample = downsample_basic_block(self.inplanes, planes*block.expansion, stride)

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

      
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResEncoder, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if cfg.resnet_relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2])
        self.proj = nn.Linear(self.backend_out, cfg.encoder_embed_dim)
        if cfg.resnet_weights is not None:
            logger.info(f"Load {cfg.resnet_weights} for resnet")
            std = torch.load(cfg.resnet_weights, map_location=torch.device('cpu'))['model_state_dict']
            frontend_std, trunk_std = OrderedDict(), OrderedDict()
            for key, val in std.items():
                new_key = '.'.join(key.split('.')[1:])
                if 'frontend3D' in key:
                    frontend_std[new_key] = val
                if 'trunk' in key:
                    trunk_std[new_key] = val
            self.frontend3D.load_state_dict(frontend_std)
            self.trunk.load_state_dict(trunk_std)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = self.threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()
        x = self.proj(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

    def threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batch*s_time, n_channels, sx, sy)


class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = None
        input_dim = cfg.audio_feat_dim
        output_dim = cfg.encoder_embed_dim
        ################################################################################
        # TODO:                                                                        #
        #  define the network                                                          #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

    def forward(self, x):
        ################################################################################
        # TODO:                                                                        #
        #  finish the forward function                                                 #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # import pdb; pdb.set_trace()

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return x