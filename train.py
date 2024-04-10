# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.crossvit import  crossvit_18_dagger_384
from models.conformer import Conformer_base_patch16
from models.swin_transformer import swin_base_patch4_window7_224
from models.volo import volo_d3
from models.focal_transformer import focal_tiny_224_window7
from models.SLCI import build_models

from utils.lr_scheduler import build_scheduler
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from timm.utils import accuracy as acc_function
from timm.utils import AverageMeter as AM
import torch.nn as nn

from logger import create_logger

from util import *
from MAP import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    print(tensor.dtype)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def save_model(args, model,logger):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, test_loader, global_step,logger):
    # Validation!
    eval_losses = AverageMeter()
    acc1_recoder = AM()
    acc5_recoder = AM()

    if args.multi_label:
        ap_meter = AveragePrecisionMeter(False)

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, only_preds, combine_preds, all_label = [], [], [], []
    if args.multi_label:
        result_ingredients = torch.Tensor().cuda()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        if args.multi_label:
            x, y, mul_cls = batch
        else:
            x, y = batch
        with torch.no_grad():
            if args.multi_label:
                final_logits, only_logits, logits, multi_logits = model(x)
                sigmoid = nn.Sigmoid()
                sig = sigmoid(multi_logits)
                tmp = sig.gt(0.5)
                #print(tmp)
                #print(np.argwhere(tmp.cpu().numpy()))
                #input()
                result_ingredients = torch.cat((result_ingredients, tmp), 0)
                #print(multi_logits.data)
                ap_meter.add(multi_logits.data, mul_cls)
            else:
                logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())


            acc1, acc5 = acc_function(logits, y, topk=(1, 5))


            preds = torch.argmax(logits, dim=-1)
            pred_only = torch.argmax(only_logits, dim=-1)
            pred_final = torch.argmax(final_logits, dim=-1)
            acc1_recoder.update(acc1.item(), y.size(0))
            acc5_recoder.update(acc5.item(), y.size(0))

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            only_preds.append(pred_only.detach().cpu().numpy())
            combine_preds.append(pred_final.detach().cpu().numpy())

            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            only_preds[0] = np.append(
                only_preds[0], pred_only.detach().cpu().numpy(), axis=0
            )
            combine_preds[0] = np.append(
                combine_preds[0], pred_final.detach().cpu().numpy(), axis=0
            )

            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
        #break


    all_preds, only_preds, combine_preds, all_label = all_preds[0], only_preds[0], combine_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    only_accuracy = simple_accuracy(only_preds, all_label)
    combine_accuracy = simple_accuracy(combine_preds, all_label)

    accuracy = torch.tensor(accuracy).to(args.device)
    only_accuracy = torch.tensor(only_accuracy).to(args.device)
    combine_accuracy = torch.tensor(combine_accuracy).to(args.device)
    dist.barrier()

    val_accuracy = reduce_mean(accuracy, args.nprocs)
    val_accuracy = val_accuracy.detach().cpu().numpy()

    on_accuracy = reduce_mean(only_accuracy, args.nprocs)
    on_accuracy = on_accuracy.detach().cpu().numpy()

    com_accuracy = reduce_mean(combine_accuracy, args.nprocs)
    com_accuracy = com_accuracy.detach().cpu().numpy()

    ap_meter.reduce_all(args.nprocs)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % val_accuracy)
    logger.info("Valid Only_Accuracy: %2.5f" % on_accuracy)
    logger.info("Valid Combine_Accuracy: %2.5f" % com_accuracy)
    logger.info("TOP-1: %2.5f" % acc1_recoder.avg)
    logger.info("TOP-5: %2.5f" % acc5_recoder.avg)
    logger.info("Multi-label Performance")
    if args.multi_label:

        ovp,ovr,ovf1,cp,cr,cf1=ap_meter.overall()
        logger.info(" OP:%2.5f \t OR:%2.5f \t OF1:%2.5f" % (ovp, ovr, ovf1))
        logger.info("CP:%2.5f \t CR:%2.5f \t CF1:%2.5f " % (cp, cr, cf1))
        
    return val_accuracy

def train(args, model, logger):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)


    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.get_config_optim(args.learning_rate, 0.1),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    t_total = args.epoch * len(train_loader) 
    warmup_steps = args.warmup_epoch * len(train_loader) 

    if args.decay_type == "cosine":
        scheduler = build_scheduler(args.epoch,args.warmup_epoch, 1e-6, 1e-7, optimizer, len(train_loader))
    else:
        scheduler = build_scheduler(args.epoch,args.warmup_epoch, 1e-6, 1e-7, optimizer, len(train_loader))

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.epoch)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    fc_losses = AverageMeter()
    mul_losses = AverageMeter()
    norms = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    num_steps = len(train_loader)
    for epoch in range(args.epoch):
        logger.info("***** Epoch: %d *****" % epoch)

        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds, all_label = [], []
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            #break
            if args.multi_label:
                x, y, mul_cls = batch
                loss, logits, mul_logits, fc_loss, mul_loss = model(x, y, mul_cls)

            else:
                x, y = batch
                loss, logits = model(x, y)

            preds = torch.argmax(logits, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                fc_losses.update(fc_loss.item()*args.gradient_accumulation_steps)
                mul_losses.update(mul_loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                norms.update(norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step_update(epoch * num_steps + step)
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                
                if global_step % args.print_freq == 0:
                    lr = optimizer.param_groups[0]['lr']
                    main_lr = optimizer.param_groups[1]['lr']

                    logger.info(
                    f'Train: [{epoch}/{args.epoch}][{step}/{len(train_loader)}]\t'
                    f'lr {lr:.6f}\t' f'main_lr {main_lr:.6f}\t'
                    f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'fc_loss{fc_losses.val:.4f} ({fc_losses.avg:.4f})\t'
                    f'mul_loss{mul_losses.val:.4f} ({mul_losses.avg:.4f})\t'
                    f'norm {norms.val:.6f} ({norms.avg:.4f}) \t')

        with torch.no_grad():
            accuracy = valid(args, model, test_loader, global_step, logger)

        if args.local_rank in [-1, 0]:
            if best_acc < accuracy:
                save_model(args, model, logger)
                best_acc = accuracy
            logger.info("best accuracy so far: %f" % best_acc)

        model.train()

        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)
        dist.barrier()
        train_accuracy = reduce_mean(accuracy, args.nprocs)
        train_accuracy = train_accuracy.detach().cpu().numpy()
        logger.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()
        if global_step % t_total == 0:
            break
        
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

def setup(args,logger):

    num_classes = 1000
    multi_label_classes = 353
    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "food101":
        num_classes = 101
        multi_label_classes = 174
    elif args.dataset == "food172":
        num_classes = 172
        multi_label_classes = 353
    elif args.dataset == "food200":
        num_classes = 200
        multi_label_classes = 399

    model = build_models(model_type = args.model_type,
                         num_class = num_classes,
                         multi_num_class = multi_label_classes,
                         pretrained = (args.pretrained_dir is None),
                         pretrained_dir = args.pretrained_dir,
                         use_cross_attention = True,
                         use_cirl = True,
                         use_icfe = True,
                         drop_path_rate = 0.3,
                         drop_path = 0.0,
                         drop = 0.0
                         )

    if args.checkpoints != "":
        checkpoints = torch.load(args.checkpoints)
        if 'model' in checkpoints.keys():
            checkpoint_model = checkpoints['model']
        else:
            checkpoint_model = checkpoints
        model.load_state_dict(checkpoint_model)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, default="sample_run",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", required=True, default="CBiAFormer-B",
                        help="which model.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017","food101","food172","food200"], default="food172",
                        help="Which dataset.")

    parser.add_argument('--data_root', type=str, default='/data5/food172')
    parser.add_argument("--pretrained_dir", type=str, default="/home/lyx/pretrained_model.pth",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")  ## 暂时不用
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=20, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=0.03, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=5e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--epoch", default=100, type=int,
                        help="Total number of training epochs to perform.") ## 
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="linear",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epoch", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")


    parser.add_argument("--local_rank", type=int, default= -1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.") ## 
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--worker_num', help='the num of workers during train and test',
                        default = 0, type=int) ## 8

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--split', type=str, default='overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    parser.add_argument('--checkpoints', type=str, default="",
                        help="the checkpoints path")
    parser.add_argument('--multi_label', type=bool, default=True,
                        help="the checkpoint path")

    args = parser.parse_args()


    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logger = create_logger(output_dir=args.output_dir, dist_rank=dist.get_rank(), name=f"Ablation_172")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args, logger)
    # Training
    train(args, model, logger)

if __name__ == "__main__":
    main()

"""
python -m torch.distributed.launch --nproc_per_node=2 train.py --fp16 --name MULTI
python -m torch.distributed.launch --nproc_per_node=2 /home/vip/xly/food_multilabel/food/train.py  --fp16 --name DMS
python -m torch.distributed.launch --nproc_per_node=2 /home/vip/xly/food_multilabel/food/train.py  --fp16 --name food172 --data_root /home/vip/lmj/scz/data/food172/images/ --data_list /home/vip/xly/data/food172/ --dataset food172
"""
