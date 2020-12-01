import torch
import config
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_chs, out_chs, opt):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_chs, 20, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 40, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm1d(num_features=40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(40, 80, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm1d(num_features=80),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv4 = nn.Sequential(
            nn.Conv1d(80, 160, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm1d(num_features=160),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv5 = nn.Sequential(
            nn.Conv1d(160, 320, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm1d(num_features=320),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.fc = nn.Linear(640, 1)

    def forward(self, x):

        # print("\nblock input size = %s" % (str(x.size())))
        x = self.conv1(x)
        # print("block out1 size = %s" % (str(x.size())))
        x = self.conv2(x)
        # print("block out2 size = %s" % (str(x.size())))
        x = self.conv3(x)
        # print("block out3 size = %s" % (str(x.size())))
        x = self.conv4(x)
        # print("block out4 size = %s" % (str(x.size())))
        x = self.conv5(x)
        # print("block out5 size = %s" % (str(x.size())))

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        # print("block x size = %s" % (str(x.size())))

        return x

class single(nn.Module):
    def __init__(self, in_chs, out_chs, opt):
        super(single, self).__init__()
        w = opt.window_size
        if opt.roi_clust == 'shen268':
            self.fc = nn.Linear(w*in_chs, 1)

        if opt.roi_clust == 'findlab90':
            self.fc = nn.Linear(w*in_chs, 1)

    def forward(self, x):
        # print("\nblock input size = %s" % (str(x.size())))

        x = x.view(x.shape[0], -1)
        print("block x size = %s" % (str(x.size())))

        x = self.fc(x)
        # print("block x size = %s" % (str(x.size())))

        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv1d(in_chs, in_chs, kernel_size=3, padding=1, groups=in_chs, bias=False)
        self.pointwise = nn.Conv1d(in_chs, out_chs, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class sepCONV1d(nn.Module):
    def __init__(self, in_chs, out_chs, opt):
        super(sepCONV1d, self).__init__()

        self.conv1 = nn.Sequential(
            depthwise_separable_conv(in_chs, 20),
            nn.MaxPool1d(kernel_size=2))

        self.conv2 = nn.Sequential(
            depthwise_separable_conv(20, 40),
            nn.MaxPool1d(kernel_size=2))

        self.conv3 = nn.Sequential(
            depthwise_separable_conv(40, 80),
            nn.MaxPool1d(kernel_size=2))

        self.conv4 = nn.Sequential(
            depthwise_separable_conv(80, 160),
            nn.MaxPool1d(kernel_size=2))

        self.conv5 = nn.Sequential(
            depthwise_separable_conv(160, 320),
            nn.MaxPool1d(kernel_size=2))

        self.fc1 = nn.Linear(320, 1)
        self.fc2 = nn.Linear(320, 1)

    def forward(self, x):
        # print("\nblock input size = %s" % (str(x.size())))
        x = self.conv1(x)
        # print("block out1 size = %s" % (str(x.size())))
        x = self.conv2(x)
        # print("block out2 size = %s" % (str(x.size())))
        x = self.conv3(x)
        # print("block out3 size = %s" % (str(x.size())))
        x = self.conv4(x)
        # print("block out4 size = %s" % (str(x.size())))
        x = self.conv5(x)
        # print("block out5 size = %s" % (str(x.size())))

        x = x.view(x.shape[0], -1)

        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2

class sepCONV1d_relu(nn.Module):
    def __init__(self, in_chs, out_chs, opt):
        super(sepCONV1d_relu, self).__init__()

        self.conv1 = nn.Sequential(
            depthwise_separable_conv(in_chs, 20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv2 = nn.Sequential(
            depthwise_separable_conv(20, 40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv3 = nn.Sequential(
            depthwise_separable_conv(40, 80),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv4 = nn.Sequential(
            depthwise_separable_conv(80, 160),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.conv5 = nn.Sequential(
            depthwise_separable_conv(160, 320),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        self.fc = nn.Linear(640, 1)

    def forward(self, x):
        # print("\nblock input size = %s" % (str(x.size())))
        x = self.conv1(x)
        # print("block out1 size = %s" % (str(x.size())))
        x = self.conv2(x)
        # print("block out2 size = %s" % (str(x.size())))
        x = self.conv3(x)
        # print("block out3 size = %s" % (str(x.size())))
        x = self.conv4(x)
        # print("block out4 size = %s" % (str(x.size())))
        x = self.conv5(x)
        # print("block out5 size = %s" % (str(x.size())))

        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x

class sepCONV1d_drop(nn.Module):
    def __init__(self, in_chs, out_chs, opt):
        super(sepCONV1d_drop, self).__init__()

        self.conv1 = nn.Sequential(
            depthwise_separable_conv(in_chs, 20),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2))

        self.conv2 = nn.Sequential(
            depthwise_separable_conv(20, 40),
            nn.Dropout(p=0.4),
            nn.MaxPool1d(kernel_size=2))

        self.conv3 = nn.Sequential(
            depthwise_separable_conv(40, 80),
            nn.Dropout(p=0.4),
            nn.MaxPool1d(kernel_size=2))

        self.conv4 = nn.Sequential(
            depthwise_separable_conv(80, 160),
            nn.Dropout(p=0.4),
            nn.MaxPool1d(kernel_size=2))

        self.conv5 = nn.Sequential(
            depthwise_separable_conv(160, 320),
            nn.Dropout(p=0.4),
            nn.MaxPool1d(kernel_size=2))

        self.fc = nn.Linear(640, 1)

    def forward(self, x):
        # print("\nblock input size = %s" % (str(x.size())))
        x = self.conv1(x)
        # print("block out1 size = %s" % (str(x.size())))
        x = self.conv2(x)
        # print("block out2 size = %s" % (str(x.size())))
        x = self.conv3(x)
        # print("block out3 size = %s" % (str(x.size())))
        x = self.conv4(x)
        # print("block out4 size = %s" % (str(x.size())))
        x = self.conv5(x)
        # print("block out5 size = %s" % (str(x.size())))

        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x

