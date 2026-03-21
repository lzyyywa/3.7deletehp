import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np

from utils.lorentz import pairwise_dist, half_aperture, oxy_angle

# =====================================================================
# 【双曲空间专属 Loss】
# =====================================================================

class HierarchicalEntailmentLoss(nn.Module):
    def __init__(self, K=0.1):
        super().__init__()
        self.K = K

    def forward(self, child, parent, c):
        with torch.cuda.amp.autocast(enabled=False):
            theta = oxy_angle(parent.float(), child.float(), curv=c.float()).unsqueeze(1)               
            alpha_parent = half_aperture(parent.float(), curv=c.float(), min_radius=self.K).unsqueeze(1) 
            loss_cone = F.relu(theta - alpha_parent)
        return loss_cone.mean()


class DiscriminativeAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, hard_weight=3.0):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_hyp, t_hyp, c, mask_pos, mask_hard):
        with torch.cuda.amp.autocast(enabled=False):
            dist = pairwise_dist(v_hyp.float(), t_hyp.float(), curv=c.float())
            logits = -dist / self.temperature
            
            B = v_hyp.size(0)
            
            if self.hard_weight > 1.0:
                logits = logits + mask_hard.float() * math.log(self.hard_weight)
                
            false_negatives = mask_pos & ~torch.eye(B, dtype=torch.bool, device=v_hyp.device)
            logits.masked_fill_(false_negatives, -1e9)
            
            labels = torch.arange(B, device=v_hyp.device)
            
        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.t(), labels)
        return (loss_v2t + loss_t2v) / 2.0


def loss_calu(predict, target, config):
    batch_img, batch_verb, batch_obj, batch_target, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda() 
    
    c_pos = predict['c_pos']
    verb_logits = predict['verb_logits']
    obj_logits = predict['obj_logits']
    
    # 获取预测字典中的组合概率
    pred_com_logits = predict['pred_com_logits']
    
    v_hyp = predict['v_hyp']                  
    o_hyp = predict['o_hyp']
    v_c_hyp = predict['v_c_hyp']              
    t_v_hyp = predict['t_v_hyp']              
    t_o_hyp = predict['t_o_hyp']
    t_c_hyp = predict['t_c_hyp']              
    coarse_v_hyp = predict['coarse_v_hyp']    
    coarse_o_hyp = predict['coarse_o_hyp']    

    ce_loss_fn = nn.CrossEntropyLoss()
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07, hard_weight=3.0)
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1)

    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)
    
    train_pairs = config.train_pairs
    train_v_inds = train_pairs[:, 0]
    train_o_inds = train_pairs[:, 1]
    pred_com_train = pred_com_logits[:, train_v_inds, train_o_inds]
    
    # 强制施加隐式的组合推理交叉熵损失
    loss_com = ce_loss_fn(pred_com_train, batch_target)

    mask_verb = (batch_verb.unsqueeze(1) == batch_verb.unsqueeze(0))
    mask_obj = (batch_obj.unsqueeze(1) == batch_obj.unsqueeze(0))
    mask_pos_comp = mask_verb & mask_obj
    mask_hard_comp = mask_verb ^ mask_obj 
    loss_dal = dal_loss_fn(v_c_hyp, t_c_hyp, c_pos, mask_pos=mask_pos_comp, mask_hard=mask_hard_comp)
    
    loss_hem_vc2vs = hem_loss_fn(child=v_c_hyp, parent=v_hyp, c=c_pos)
    loss_hem_vc2vo = hem_loss_fn(child=v_c_hyp, parent=o_hyp, c=c_pos)
    loss_hem_tc2ts = hem_loss_fn(child=t_c_hyp, parent=t_v_hyp, c=c_pos)
    loss_hem_tc2to = hem_loss_fn(child=t_c_hyp, parent=t_o_hyp, c=c_pos)
    loss_hem_ts2tsp = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    loss_hem_to2top = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)

    loss_hem = loss_hem_vc2vs + loss_hem_vc2vo + \
               loss_hem_tc2ts + loss_hem_tc2to + \
               loss_hem_ts2tsp + loss_hem_to2top

    w_cls = getattr(config, 'w_cls', 1.0)
    w_com = getattr(config, 'w_com', 1.0)
    w_dal = getattr(config, 'w_dal', 1.0)
    w_hem = getattr(config, 'w_hem', 1.0)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_com * loss_com + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem
                 
    loss_dict = {
        'loss_cls_verb': loss_cls_verb.item(),
        'loss_cls_obj': loss_cls_obj.item(),
        'loss_com': loss_com.item(),
        'loss_dal': loss_dal.item(),
        'loss_hem': loss_hem.item()
    }

    return total_loss, loss_dict


# =====================================================================
# 【原欧式项目中的通用辅助 Loss 工具类 (从 eur_c2c 补齐)】
# =====================================================================

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    """
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label, mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss * batch_size
        else:
            return loss


def hsic_loss(input1, input2, unbiased=False):
    def _kernel(X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    # we simply use the squared dimension of feature as the sigma for RBF kernel
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    # compute the kernels
    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic / (N * (N - 3))
    else:
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


class Gml_loss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    Loss from No One Left Behind: Improving the Worst Categories in Long-Tailed Learning
    """
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()

    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        '''
        Args:
            p_o_on_v: b,n_v,n_o
            o_label: b,
            n_c: b,n_o
        '''
        n_c = n_c[:, 0]
        b = p_o_on_v.shape[0]
        n_o = p_o_on_v.shape[-1]
        p_o = p_o_on_v[range(b), v_label, :]  # b,n_o

        num_c = n_c.sum().view(1, -1)  # 1,n_o

        p_o_exp = torch.exp(p_o * t)
        p_o_exp_wed = num_c * p_o_exp  # b,n_o
        p_phi = p_o_exp_wed / torch.sum(p_o_exp_wed, dim=0, keepdim=True)  # b,n_o

        p_ba = torch.sum(p_phi * n_c, dim=0, keepdim=True) / (num_c + 1.0e-6)  # 1,n_o
        p_ba[torch.where(p_ba < 1.0e-8)] = 1.0
        p_ba_log = torch.log(p_ba)
        loss = (-1.0 / n_o) * p_ba_log.sum()

        return loss