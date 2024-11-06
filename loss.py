import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        label_shape = x.shape
        label = torch.ones(label_shape[0], device=x.device)  # Note: device should match input device
        generator_loss = self.BCELoss(x, label)
        return generator_loss


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, real_output, fake_output):
        real_output = torch.squeeze(real_output, dim=1)
        fake_output = torch.squeeze(fake_output, dim=1)
        label_shape = real_output.shape
        label_T = torch.ones(label_shape[0], device=real_output.device)
        label_F = torch.zeros(label_shape[0], device=fake_output.device)
        loss_DT = self.BCELoss(real_output, label_T)
        loss_DF = self.BCELoss(fake_output, label_F)
        dis_loss = 1 * (loss_DT + loss_DF)

        return dis_loss


class GAN_loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.Dis = DiscriminatorLoss()
        self.Gen = GeneratorLoss()

    def forward(self, real_output, fake_output):
        real_output = torch.squeeze(real_output, dim=1)
        fake_output = torch.squeeze(fake_output, dim=1)
        label_shape = real_output.shape
        label_T = torch.ones(label_shape[0], device=real_output.device)
        label_F = torch.zeros(label_shape[0], device=fake_output.device)
        loss_DT = self.BCELoss(real_output, label_T)
        loss_DF = self.BCELoss(fake_output, label_F)
        dis_loss = 0.5 * (loss_DT + loss_DF)

        return dis_loss