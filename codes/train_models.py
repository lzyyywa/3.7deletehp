import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
import torch.multiprocessing
import numpy as np
import json
import math
from torch.nn import CrossEntropyLoss
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter
from utils.hsic import hsic_normalized_cca


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    
    # 欧式独有的损失函数实例
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    # 欧式需要的训练对索引
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []
    
    # 获取消融实验开关 (默认 True 走双曲)
    use_hyperbolic = getattr(config, 'use_hyperbolic', True)

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_cls_v_losses = []
        epoch_cls_o_losses = []
        
        # 彻底分离：双曲专属监控列表 vs 欧式专属监控列表
        epoch_dal_losses = []
        epoch_hem_losses = []
        epoch_com_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        
        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            
            # 兼容不同 dataset 输出长度
            if len(batch) > 4:
                batch_coarse_verb = batch[4].cuda()
                batch_coarse_obj = batch[5].cuda()
                target = [batch_img, batch_verb, batch_obj, batch_target, batch_coarse_verb, batch_coarse_obj]
            else:
                batch_coarse_verb = None
                batch_coarse_obj = None
                target = [batch_img, batch_verb, batch_obj, batch_target]

            with torch.cuda.amp.autocast(enabled=True):
                # 调用统一的模型前向传播
                predict = model(
                    video=batch_img,
                    batch_verb=batch_verb,
                    batch_obj=batch_obj,
                    batch_coarse_verb=batch_coarse_verb,
                    batch_coarse_obj=batch_coarse_obj,
                    pairs=batch_target
                )

                if use_hyperbolic:
                    # ====================================================
                    # 【双曲模式分支】
                    # ====================================================
                    loss, loss_dict = loss_calu(predict, target, config)
                    loss = loss / config.gradient_accumulation_steps
                    
                    # 记录双曲独有的损失
                    epoch_dal_losses.append(loss_dict.get('loss_dal', 0.0))
                    epoch_hem_losses.append(loss_dict.get('loss_hem', 0.0))
                    
                else:
                    # ====================================================
                    # 【欧式模式分支】
                    # ====================================================
                    p_v = predict['verb_logits_euc']
                    p_o = predict['obj_logits_euc']
                    f = predict['pred_com_euc']

                    # 提取合法组合
                    train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                    pred_com_train = f[:, train_v_inds, train_o_inds]
                    
                    cosine_scale = getattr(config, 'cosine_scale', 100.0)

                    # 计算交叉熵
                    loss_com = Loss_fn(pred_com_train * cosine_scale, batch_target)
                    loss_verb = Loss_fn(p_v * cosine_scale, batch_verb)
                    loss_obj = Loss_fn(p_o * cosine_scale, batch_obj)

                    total_loss = loss_com + 0.2 * (loss_verb + loss_obj)
                    loss = total_loss / config.gradient_accumulation_steps
                    
                    # 生成名称完全匹配概念的欧式字典
                    loss_dict = {
                        'loss_cls_verb': loss_verb.item(),
                        'loss_cls_obj': loss_obj.item(),
                        'loss_com': loss_com.item()
                    }
                    
                    # 记录欧式独有的损失
                    epoch_com_losses.append(loss_dict['loss_com'])

            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 记录总损失和基元损失
            epoch_train_losses.append(loss.item() * config.gradient_accumulation_steps)
            epoch_cls_v_losses.append(loss_dict['loss_cls_verb'])
            epoch_cls_o_losses.append(loss_dict['loss_cls_obj'])

            # ================= 根据模式打印进度条 =================
            if use_hyperbolic:
                current_c = predict['c_pos'].item()
                if hasattr(model, 'module'): 
                    current_temp = F.softplus(model.module.cls_temp).item() + 0.05
                else:
                    current_temp = F.softplus(model.cls_temp).item() + 0.05

                progress_bar.set_postfix({
                    "loss": f"{np.mean(epoch_train_losses[-50:]):.2f}",
                    "v_cls": f"{np.mean(epoch_cls_v_losses[-50:]):.2f}",
                    "o_cls": f"{np.mean(epoch_cls_o_losses[-50:]):.2f}",
                    "dal": f"{np.mean(epoch_dal_losses[-50:]):.2f}",
                    "hem": f"{np.mean(epoch_hem_losses[-50:]):.2f}",
                    "c": f"{current_c:.3f}", 
                    "tau": f"{current_temp:.3f}"
                })
            else:
                progress_bar.set_postfix({
                    "loss": f"{np.mean(epoch_train_losses[-50:]):.2f}",
                    "vv": f"{np.mean(epoch_cls_v_losses[-50:]):.2f}",
                    "oo": f"{np.mean(epoch_cls_o_losses[-50:]):.2f}",
                    "com": f"{np.mean(epoch_com_losses[-50:]):.2f}"  # 彻底显示 com
                })
            
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()
        
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses):.4f}")
        train_losses.append(np.mean(epoch_train_losses))
        
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} cls_verb loss {np.mean(epoch_cls_v_losses):.4f}\n")
        log_training.write(f"epoch {i + 1} cls_obj loss {np.mean(epoch_cls_o_losses):.4f}\n")
        
        # ================= 根据模式写入 log =================
        if use_hyperbolic:
            log_training.write(f"epoch {i + 1} dal loss {np.mean(epoch_dal_losses):.4f}\n")
            log_training.write(f"epoch {i + 1} hem loss {np.mean(epoch_hem_losses):.4f}\n")
        else:
            log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses):.4f}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
            
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()
        
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))