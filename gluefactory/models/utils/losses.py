import torch
import torch.nn as nn
from omegaconf import OmegaConf


def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        """
        Constructs the supervision weights for the negative log-likelihood loss.
        It computes the weights on the fly from the padded 1D match vectors.
        """
        
        # log_assignment has shape [B, M, N], where M and N are padded lengths
        # data tensors like gt_matches0 are also padded, e.g., shape [B, M]
        b, m, n = log_assignment.shape
        device = log_assignment.device
    
        # Get the padded ground truth matches
        gt_matches0 = data["gt_matches0"] # Shape: [B, M]
        gt_matches1 = data["gt_matches1"] # Shape: [B, N]

        # Create the positive assignment matrix
        # We want a [B, M, N] matrix where entry (b, i, j) is 1 if kpt i in image 0
        # of batch element b matches kpt j in image 1.
        positive = torch.zeros_like(log_assignment)

        # Find where the valid matches are (i.e., not -1 padding from dataset,
        # and not 0 padding from collate_fn)
        valid0 = (gt_matches0 > -1)
        
        # Get batch and source indices for all valid matches
        b_idx, src_idx = torch.where(valid0)
        
        # Get the corresponding target indices
        tgt_idx = gt_matches0[valid0]

        # Use these indices to populate the positive matrix
        # This is equivalent to: for b, i, j in zip(b_idx, src_idx, tgt_idx): positive[b, i, j] = 1
        positive[b_idx, src_idx, tgt_idx] = 1.0

        # Negative weights for dustbins (unmatched points)
        # These are already padded correctly by the collate function
        neg0 = (gt_matches0 == -1).float() # Unmatched in image 0
        neg1 = (gt_matches1 == -1).float() # Unmatched in image 1

        # Construct final weights tensor
        weights = torch.zeros_like(log_assignment)
        
        # Positive weights for the [M, N] assignment part
        weights[:, :m, :n] = positive

        # Assign negative weights to the dustbin slots.
        # The slice must be [:, :m-1] to match neg0's shape [B, M_pad]
        weights[:, :m-1, -1] = neg0
        weights[:, -1, :n-1] = neg1

        # We can also choose to ignore padding indices in the loss calculation.
        # The keypoint tensors are padded with 0, so we can get a mask.
        # This ensures we don't penalize predictions on padded (fake) keypoints.
        if 'keypoints0' in data:
            valid_mask0 = (data['keypoints0'].abs().sum(dim=-1) > 0) # [B, M]
            valid_mask1 = (data['keypoints1'].abs().sum(dim=-1) > 0) # [B, N]
            
            # Apply mask to the dustbin weights
            weights[:, :m-1, -1] *= valid_mask0
            weights[:, -1, :n-1] *= valid_mask1
            
            # Also apply to the main assignment matrix, Slice to [:m-1, :n-1] to match the dimensions of the valid masks
            weights[:, :m-1, :n-1] *= valid_mask0.unsqueeze(-1) * valid_mask1.unsqueeze(-2)

        return weights
    
        # m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        # positive = data["gt_assignment"].float()
        # neg0 = (data["gt_matches0"] == -1).float()
        # neg1 = (data["gt_matches1"] == -1).float()

        # weights = torch.zeros_like(log_assignment)
        # weights[:, :m, :n] = positive

        # weights[:, :m, -1] = neg0
        # weights[:, -1, :n] = neg1
        # return weights
