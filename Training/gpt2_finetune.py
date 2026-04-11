import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from Definitions.Model import *
import json
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import subprocess

# region argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('cross_val_counter', type=int)
argparser.add_argument('language', type=str)
argparser.add_argument('task', type=str)
argparser.add_argument('data_path', type=str)
cli_args = argparser.parse_args()
# endregion argparse

args = Hyperparameters()

args.input_bin     = os.path.join(cli_args.data_path, f"{cli_args.task}_{cli_args.language}_gpt2_train_{cli_args.cross_val_counter}_*.bin")
args.input_val_bin = os.path.join(cli_args.data_path, f"{cli_args.task}_{cli_args.language}_gpt2_val_{cli_args.cross_val_counter}_*.bin")
args.batch_size      = 8
args.sequence_length = 64
args.num_iterations  = 256
args.val_loss_every  = 2
args.save_every      = 32
args.val_tokens      = 1920

# DDP setup
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank       = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)

# Logging
logfile = None
if master_process:
    run_id  = f"{cli_args.language}_gpt2_{cli_args.task}_{cli_args.cross_val_counter}"
    logdir  = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id

def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s + '\n')

print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('=' * 100, logonly=True)

# Batch / step sizing
B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# Data loaders
train_loader = DistributedDataLoader(args.input_bin,     B, T, ddp_rank, ddp_world_size)
val_loader   = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('=' * 100, logonly=True)
x, y = train_loader.next_batch()

# Load GPT-2 and wrap for training
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.cuda().bfloat16()
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

# Optimizers
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.006,   betas=(0.8, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=0.00008, betas=(0.8, 0.95), fused=True)
params        = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2]
optimizer3    = Muon(matrix_params, lr=0.0005, momentum=0.95)
optimizer4    = torch.optim.Adam(scalar_params, lr=0.0004, betas=(0.8, 0.95), fused=True)
optimizers    = [optimizer1, optimizer2, optimizer3, optimizer4]

def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.cooldown_iters

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# -----------------------------------------------------------------------------
# Training loop
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # Validation
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(input_ids=x_val.unsqueeze(0), labels=y_val.unsqueeze(0)).loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if master_process:
            print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # Training
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        if i < train_accumulation_steps:
            with model.no_sync():
                loss = model(input_ids=x.unsqueeze(0), labels=y.unsqueeze(0)).loss
                x, y = train_loader.next_batch()
                loss.backward()
        else:
            loss = model(input_ids=x.unsqueeze(0), labels=y.unsqueeze(0)).loss
            x, y = train_loader.next_batch()
            loss.backward()
    train_loss = loss.detach()

    for p in model.parameters():
        p.grad /= train_accumulation_steps
    frac = min(step / 300, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

# -----------------------------------------------------------------------------
# Evaluation (master process only, using trained raw_model)
if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    COUNT_MIN_ID = 50257 - 29  # 50227
    COUNT_MAX_ID = 50257

    pattern = os.path.join(
        cli_args.data_path,
        f"{cli_args.task}_{cli_args.language}_gpt2_test_{cli_args.cross_val_counter}_000000.bin"
    )
    with open(pattern, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)

    out_dir   = "finetune_eval"
    os.makedirs(out_dir, exist_ok=True)
    out_file  = os.path.join(out_dir, f"{cli_args.task}_{cli_args.language}_gpt2_{cli_args.cross_val_counter}.txt")
    meta_file = os.path.join(out_dir, f"{cli_args.task}_{cli_args.language}_gpt2_{cli_args.cross_val_counter}_meta.txt")

    raw_model.eval()

    if cli_args.task in ["syllables", "word"]:
        separators = (tokens >= COUNT_MIN_ID) & (tokens <= COUNT_MAX_ID)
        sep_idx    = np.where(separators)[0]
        splits     = np.split(tokens, sep_idx + 1)

        hits = 0
        tot  = 0

        with open(meta_file, "w") as meta:
            for s in splits:
                if len(s) <= 3:
                    continue

                inp    = torch.tensor(s[1:-1], dtype=torch.long).unsqueeze(0).to(device)
                target = int(s[-1])

                with torch.inference_mode():
                    outputs           = raw_model(input_ids=inp)
                    logits            = outputs.logits
                    last_token_logits = logits[0, -1].clone()
                    last_token_logits[:COUNT_MIN_ID]     = float('-inf')
                    last_token_logits[COUNT_MAX_ID + 1:] = float('-inf')
                    pred = torch.argmax(last_token_logits).item()

                correct = (pred == target)
                hits   += int(correct)
                tot    += 1
                meta.write(json.dumps({
                    "input":      inp.squeeze(0).tolist(),
                    "target":     target,
                    "prediction": pred,
                    "correct":    correct,
                }) + "\n")

        acc  = hits / tot if tot > 0 else 0.0
        line = f"{cli_args.task},{cli_args.language},gpt2,{cli_args.cross_val_counter},{acc:.6f},{hits},{tot}\n"
        print(f"Accuracy: {acc:.4f} ({hits}/{tot})")
        with open(out_file, "a") as f:
            f.write(line)

    else:  # G2P
        eos_id = COUNT_MIN_ID
        splits = np.split(tokens, np.where(tokens == eos_id)[0])

        hits = 0
        tot  = 0

        with open(meta_file, "w") as meta:
            for s in splits:
                split_idx = np.where(s == eos_id)[0]
                if len(split_idx) == 0:
                    continue
                split_idx = split_idx[0]
                inp    = s[:split_idx]
                actual = s[split_idx + 1:]

                generated     = []
                prompt_tokens = inp.tolist()

                with torch.inference_mode():
                    for _ in range(25):
                        tokens_tensor = torch.tensor(
                            prompt_tokens, dtype=torch.long, device=device
                        ).unsqueeze(0)
                        outputs       = raw_model(input_ids=tokens_tensor)
                        next_token_id = torch.argmax(outputs.logits[0, -1]).item()
                        generated.append(next_token_id)
                        if next_token_id == eos_id:
                            break
                        prompt_tokens.append(next_token_id)

                actual_trimmed = [int(x) for x in actual   if x != eos_id]
                gen_trimmed    = [int(x) for x in generated if x != eos_id]
                correct        = gen_trimmed == actual_trimmed
                hits += int(correct)
                tot  += 1
                meta.write(json.dumps({
                    "input":     inp.tolist(),
                    "target":    actual_trimmed,
                    "generated": gen_trimmed,
                    "correct":   correct,
                }) + "\n")

        acc  = hits / tot if tot > 0 else 0.0
        line = f"{cli_args.task},{cli_args.language},gpt2,{cli_args.cross_val_counter},{acc:.6f},{hits},{tot}\n"
        print(f"Accuracy: {acc:.4f} ({hits}/{tot})")
        with open(out_file, "a") as f:
            f.write(line)

dist.destroy_process_group()