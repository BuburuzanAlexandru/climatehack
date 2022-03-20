import torch
import torch.nn as nn

get_activation = {
    'leaky': nn.LeakyReLU(inplace=True),
    'relu': nn.ReLU(inplace=True),
    'sigmoid': nn.Sigmoid()
}

config_3x3_16_3x3_32_3x3_64 = {
    'encoder' : [('convlstm', '', 1, 8, 3, 1, 1),
                 ('down', '', 8, 16, 0, 0, 0),
                 ('convlstm', '', 16, 16, 3, 1, 1),
                 ('down', '', 16, 32, 0, 0, 0),
                 ('convlstm', '', 32, 32, 3, 1, 1),
                 ('down', '', 32, 64, 0, 0, 0),
                 ('convlstm', '', 64, 64, 3, 1, 1)],
    'decoder' : [('convlstm', '', 64, 64, 3, 1, 1, False),
                 ('up', '', 64, 32, 0, 0, 0, True),
                 ('convlstm', '', 64, 32, 3, 1, 1, False),
                 ('up', '', 32, 16, 0, 0, 0, True),
                 ('convlstm', '', 32, 16, 3, 1, 1, False),
                 ('up', '', 16, 8, 0, 0, 0, True),
                 ('convlstm', '', 16, 8, 3, 1, 1, False),
                 ('conv', 'relu', 8, 8, 3, 1, 1, True),
                 ('conv', 'sigmoid', 8, 1, 1, 0, 1, False)]
}


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels//2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super(ConvLSTMBlock, self).__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)) #try without

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:, t], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous(), hx, cx # (S, B, C, H, W) -> (B, S, C, H, W)


class ConvLSTMBlockDecoder(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1, forecast_steps=24, use_bn=True):
        super(ConvLSTMBlockDecoder, self).__init__()
        self.num_features = num_features
        self.forecast_steps = forecast_steps
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                     kernel_size, padding, stride, use_bn)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride, use_bn=True):
        if use_bn:
            return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                           kernel_size, padding=padding, stride=stride, bias=False),
                                 nn.BatchNorm2d(out_channels)) #try without

        return nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

    def forward(self, context, hx, cx, inputs=None):
        outputs = []
        for t in range(self.forecast_steps):
            if inputs is None:
                combined = torch.cat([context, hx], dim=1)
            else:    
                combined = torch.cat([inputs[:, t], context, hx], dim=1)
            
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config['encoder']):
            setattr(self, params[0] + '_' + str(idx), self._make_layer(*params))
            self.layers.append(params[0] + '_' + str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(get_activation[activation])
        elif type == 'convlstm':
            return ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        elif type == 'down':
            return Down(in_ch, out_ch)

        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        outputs = []
        context_h, context_c = [], []
        for layer in self.layers:
            if 'conv_' in layer or 'down' in layer:
                out = [getattr(self, layer)(x[:, i, ...]) for i in range(x.shape[1])]
                x = torch.stack(out, dim=1)

            if 'convlstm' in layer:
                x, hx, cx = getattr(self, layer)(x)
                outputs.append(x[:, -1])
                context_h.append(hx)
                context_c.append(cx)

        return outputs, context_h, context_c

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = []
        for idx, params in enumerate(config['decoder']):
            setattr(self, params[0] + '_' + str(idx), self._make_layer(*params))
            self.layers.append(params[0] + '_' + str(idx))

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size, padding, stride, use_bn=False):
        layers = []
        if type == 'conv':
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            if use_bn: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(get_activation[activation])
        elif type == 'convlstm':
            return ConvLSTMBlockDecoder(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, use_bn=use_bn)
        elif type == 'deconv':
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False))
            if use_bn: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(get_activation[activation])
        elif type == 'up':
            return Up(in_ch, out_ch)

        return nn.Sequential(*layers)

    def forward(self, encoder_outputs, context_h, context_c):
        '''
        :param x: (B, S, C, H, W)
        :return:
        '''
        idx = len(encoder_outputs)-1
        for layer in self.layers:
            if 'conv_' in layer or 'deconv_' in layer or 'up' in layer:
                out = [getattr(self, layer)(x[:, i, ...]) for i in range(x.shape[1])]
                x = torch.stack(out, dim=1)
            elif 'convlstm' in layer:
                if '0' in layer:
                    x = getattr(self, layer)(encoder_outputs[idx], context_h[idx], context_c[idx])
                else:
                    x = getattr(self, layer)(encoder_outputs[idx], context_h[idx], context_c[idx], inputs=x)
                idx -= 1
        return x

class ConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        config = config_3x3_16_3x3_32_3x3_64
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        x, context_h, context_c = self.encoder(x)
        x = self.decoder(x, context_h, context_c)
        return x[:, :, :, 32: 96, 32: 96] * 1024