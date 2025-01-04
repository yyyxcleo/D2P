import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class WKDLoss(nn.Module):
    def __init__(
        self,
        temperature=4,
        alpha_kd=1,
        beta_kd=1,
        **kwargs
        ):
        super().__init__()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
    
    def forward(self, logits_tea, logits_stu, groups=None, **kwargs):
        vanilla_kd = kwargs.get('vanilla_kd', True)
        y_tea_inter = F.softmax(
                logits_tea / self.temperature, dim=1
            )
        y_stu_inter= F.softmax(
                logits_stu / self.temperature, dim=1
            )
        if groups is None:
            y_stu_inter_log = torch.log(y_stu_inter)
            return (F.kl_div(
                    y_stu_inter_log,
                    y_tea_inter,
                    reduction='none'
                ).sum(1) * (self.temperature ** 2)).mean()
        else:
            y_tea_inter = self.cat_mask(y_tea_inter, groups)
            y_stu_inter = self.cat_mask(y_stu_inter, groups)
            y_stu_inter_log = torch.log(y_stu_inter)
            
            loss_inter = (
                    F.kl_div(y_stu_inter_log, y_tea_inter, reduction='none').sum(1).mean(0)
                    * (self.temperature**2)
                    )
            loss_inter = loss_inter * self.alpha_kd

            loss_intra = 0
            for i, g in enumerate(groups):
                y_tea_intra = F.softmax(
                        logits_tea[:, g] / self.temperature, dim=1
                    )
                y_stu_intra_log = F.log_softmax(
                        logits_stu[:, g] / (self.temperature), dim=1
                    )

                loss_intra += (
                        y_tea_inter[:, i] * F.kl_div(y_stu_intra_log, y_tea_intra, reduction='none').sum(1)
                        * (self.temperature**2)
                    ).mean()
        
        return loss_inter + loss_intra
    
    def cat_mask(self, t, groups):
        to_cat = []
        for g in groups:
            to_cat.append(t[:, g].sum(dim=1, keepdims=True))
        rt = torch.cat(to_cat, dim=1)
        return rt