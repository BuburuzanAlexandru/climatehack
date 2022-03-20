import torch
import torch.nn as nn
import antialiased_cnns

get_activation = {
    'leaky': nn.LeakyReLU(inplace=True),
    'relu': nn.ReLU(inplace=True),
    'sigmoid': nn.Sigmoid(),
    'swish': nn.SiLU(inplace=True)
}

# in_chan, out_chan, kernel, padding, stride, use_batch_norm
# net_config = {
#     'encoder' : [('convlstm', 'swish', 1, 32, 3, 1, 1),
#                  ('down', 'swish', 32, 64),
#                  ('convlstm', '', 64, 64, 3, 1, 1),
#                  ('down', 'swish', 64, 128),
#                  ('convlstm', '', 128, 128, 3, 1, 1),
#                  ('down', 'swish', 128, 256),
#                  ('convlstm', '', 256, 256, 3, 1, 1)],
#     'decoder' : [('convlstm', '', 256, 256, 3, 1, 1),
#                  ('up', 'swish', 256, 128),
#                  ('convlstm', '', 256, 128, 3, 1, 1),
#                  ('up', 'swish', 128, 64),
#                  ('convlstm', '', 128, 64, 3, 1, 1),
#                  ('up', 'swish', 64, 32),
#                  ('convlstm', '', 64, 32, 3, 1, 1),
#                  ('conv', 'swish', 32, 16, 3, 1, 1),
#                  ('conv', 'sigmoid', 16, 1, 1, 0, 1, False)]
# }

net_config = {
    'encoder' : [('convlstm', 'swish', 1, 32, 3, 1, 1),
                 ('down', 'swish', 32, 64),
                 ('convlstm', '', 64, 64, 3, 1, 1),
                 ('down', 'swish', 64, 128),
                 ('convlstm', '', 128, 128, 3, 1, 1)],
    'decoder' : [('convlstm', '', 128, 128, 3, 1, 1),
                 ('up', 'swish', 128, 64),
                 ('convlstm', '', 128, 64, 3, 1, 1),
                 ('up', 'swish', 64, 32),
                 ('convlstm', '', 64, 32, 3, 1, 1),
                 ('conv', 'swish', 32, 16, 3, 1, 1),
                 ('conv', 'sigmoid', 16, 1, 1, 0, 1, False)]
}



def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Down(nn.Module):
    """Residual downscaling block using strided convolutions"""

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        # conv1x1->blurpool is the same as blurpool->conv1x1; the latter is cheaper
        self.downsample = nn.Sequential(
            antialiased_cnns.BlurPool(in_channels, stride=2),
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.blur = antialiased_cnns.BlurPool(out_channels, stride=2)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = get_activation[activation]


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.blur(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # make skip connection
        identity = self.downsample(x)
        out += identity

        out = self.activation(out)

        return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = conv3x3(out_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = get_activation[activation]


    def forward(self, x):
        x = self.up_conv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        return x


class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super(ConvLSTMBlock, self).__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(in_channels+num_features, num_features*4, kernel_size, padding=padding, stride=stride, bias=False)

    def forward(self, inputs):
        '''
        :param inputs: (B, S, C, H, W)
        :return:
        '''
        outputs = []
        B, S, C, H, W = inputs.shape

        # create hidden and cell state
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)

        for t in range(S):
            combined = torch.cat([inputs[:, t], hx], dim=1)
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

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous(), hx, cx


class ConvLSTMBlockDecoder(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super(ConvLSTMBlockDecoder, self).__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(in_channels+num_features, num_features*4, kernel_size, padding=padding, stride=stride, bias=False)

    def forward(self, context, hx, cx, forecast_steps, inputs=None):
        outputs = []
        for t in range(forecast_steps):
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

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size=None, padding=None, stride=None, use_bn=True):
        if type == 'conv':
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
                get_activation[activation]
            )
        elif type == 'convlstm':
            return ConvLSTMBlock(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        elif type == 'down':
            return Down(in_ch, out_ch, activation)

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

    def _make_layer(self, type, activation, in_ch, out_ch, kernel_size=None, padding=None, stride=None, use_bn=True):
        if type == 'conv':
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
                get_activation[activation]
            )
        elif type == 'convlstm':
            return ConvLSTMBlockDecoder(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        elif type == 'up':
            return Up(in_ch, out_ch, activation)

    def forward(self, encoder_outputs, context_h, context_c, forecast_steps):
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
                    x = getattr(self, layer)(encoder_outputs[idx], context_h[idx], context_c[idx], forecast_steps)
                else:
                    x = getattr(self, layer)(encoder_outputs[idx], context_h[idx], context_c[idx], forecast_steps, inputs=x)
                idx -= 1
        return x


class ConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        config = net_config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, crop=True, forecast_steps=24):
        x, context_h, context_c = self.encoder(x)
        x = self.decoder(x, context_h, context_c, forecast_steps)

        if crop:
            x = x[:, :, :, 32: 96, 32: 96]

        return x * 1024