import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self):
        super(NTXentLoss, self).__init__()

    def forward(self, x, pos_indices, temperature):
        assert len(x.size()) in [2, 4]  # Input can be either 2D or 4D
        if len(x.size()) == 4:
            # Assuming x is a 4D tensor [batch_size, channels, height, width]
            x = torch.mean(x, dim=(2, 3))
        # Add indexes of the principal diagonal elements to pos_indices
        pos_indices = torch.cat([
            pos_indices,
            torch.arange(x.size(0)).reshape(x.size(0), 1).expand(-1, 2).to(pos_indices.device),
        ], dim=0)
        pos_indices = pos_indices.to(x.device)
        # Ground truth labels
        # Ground truth labels
        target = torch.zeros(x.size(0), x.size(0), device=x.device)
        pos_indices = pos_indices.long().to(device=x.device)
        target[pos_indices[:, 0], pos_indices[:, 1]] = 1.0

        # Cosine similarity
        xcs = F.cosine_similarity(x[None, :, :].to(x.device), x[:, None, :].to(x.device), dim=-1)
        # Set logit of diagonal element to "inf" signifying complete
        # correlation. sigmoid(inf) = 1.0 so this will work out nicely
        # when computing the Binary cross-entropy Loss.
        # 在计算 xcs 之后添加以下调试代码

        xcs[torch.eye(x.size(0)).bool()] = float("inf")

        # Standard binary cross-entropy loss.
        loss = F.binary_cross_entropy((xcs / temperature).sigmoid(), target, reduction="none")

        target_pos = target.bool()
        target_neg = ~target_pos
        loss = loss.to(device=x.device)
        target_pos = target_pos.to(device=x.device)
        target_neg = target_neg.to(device=x.device)

        loss_pos = torch.zeros_like(loss).masked_scatter(target_pos, loss[target_pos])
        loss_neg = torch.zeros_like(loss).masked_scatter(target_neg, loss[target_neg])
        loss_pos = loss_pos.sum(dim=1).to(device=x.device)
        loss_neg = loss_neg.sum(dim=1).to(device=x.device)
        num_pos = target.sum(dim=1).to(device=x.device)
        num_neg = x.size(0) - num_pos.to(device=x.device)

        return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()



