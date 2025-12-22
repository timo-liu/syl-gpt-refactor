# region imports
import os
import time
import smtplib
from email.message import EmailMessage
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import wandb
from Definitions.Model import *
from dotenv import load_dotenv
from dataclasses import dataclass
# endregion imports

# region email
def send_email_gmail(sender: str, recipient: str, subject: str, body: str, app_password: str):
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, app_password)      # use App Password here
        smtp.send_message(msg)
# endregion email

# region dotenv
load_dotenv()
email_password = os.getenv("EMAIL_PASSWORD")
wandb_api_key = os.getenv("WANDB_API_KEY")
# endregion dotenv

# region argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('config', type=str)
argparser.add_argument('data_path', type=str)
argparser.add_argument('out_path', type=str)
argparser.add_argument('--weights_path', type=str)
argparser.add_argument('--pretraining', type=bool, default=False)
argparser.add_argument('--task', type=str)
argparser.add_argument('--cross_val_counter', type=int)
cli_args = argparser.parse_args()
# endregion argparse

config = GPTConfig.load(cli_args.config)
args = Hyperparameters()

# set vocab to next multiple of 128
def next_multiple_of_128(v: int) -> int:
    if v <= 0:
        return 128  # or 0 if you want non-positive numbers to map differently
    return ((v + 127) // 128) * 128

config.vocab_size = next_multiple_of_128(config.vocab_size)

if cli_args.pretraining:
    print("Pretraining")
    args.input_bin = f"{config.language}_{config.paradigm}_CORPUS/{config.language}_{config.paradigm}_train_*.bin"
    args.input_val_bin = f"{config.language}_{config.paradigm}_CORPUS/{config.language}_{config.paradigm}_val_*.bin"
else:
    args.input_bin = f"{cli_args.task}_{config.language}_{config.paradigm}_train_{cli_args.cross_val_counter}_*.bin"
    args.input_val_bin = f"{cli_args.task}_{config.language}_{config.paradigm}_val_{cli_args.cross_val_counter}_*.bin"
    args.batch_size = 8
    args.sequence_length = 64
    args.num_iterations = 256
    args.val_loss_every = 2
    args.val_tokens = 1920

args.input_bin = os.path.join(cli_args.data_path, args.input_bin)
args.input_val_bin = os.path.join(cli_args.data_path, args.input_val_bin)

# -----------------------------------------------------------------------------
# int main

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    wandb.login(key=wandb_api_key)
    wandb.init(project="tokenization_tests", name=f"{config.language}_{config.paradigm}_{cli_args.pretraining}_{cli_args.task}_{cli_args.cross_val_counter}")
    run_id = f"{config.language}_{config.paradigm}_{cli_args.pretraining}_{cli_args.task}_{cli_args.cross_val_counter}"
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
def print0(s, logonly=False):
    if master_process:
        with open(logfile, "a") as f:
            if not logonly:
                print(s)
            f.write(s+'\n')
# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print0('='*100, logonly=True)
x, y = train_loader.next_batch()

model = GPT(GPTConfig(vocab_size=config.vocab_size, n_layer=12, n_head=6, n_embd=768))

if not cli_args.pretraining:
    state_dict = torch.load(cli_args.weights_path)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in torch.load(cli_args.weights_path, map_location="cpu")["model"].items()})
    print("loaded model")

model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(True)
enable_flash_sdp(False)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)

# init the optimizer(s)
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.6 if cli_args.pretraining else 0.006,   betas=(0.8, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight],         lr=0.008 if cli_args.pretraining else 0.00008, betas=(0.8, 0.95), fused=True)
params = list(raw_model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
optimizer3 = Muon(matrix_params, lr=0.05 if cli_args.pretraining else 0.0005, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04 if cli_args.pretraining else 0.0004, betas=(0.8, 0.95), fused=True) # note that this learning rate is neither sensitive nor tuned
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Set the attention blocksize for the current step, in chunks of 64. By @fernbear.bsky.social
    attn_blocksize = torch.tensor(64*((step/args.num_iterations * (1792 - 64) + 64)//64), dtype=torch.int, device='cuda')

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize, eos = 286 if (config.paradigm == "syl") else 288)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if master_process:
            wandb.log({'valloss': val_loss})
        # log val loss to console and to logfile
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        # log = dict(step=step, code="", model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        # torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # if cli_args.pretraining:
        #     torch.save(log, os.path.join(cli_args.out_path, f"{config.language}_{config.paradigm}_{step}.pth"))
        # else:
        #     torch.save(log, os.path.join(cli_args.out_path, f"{config.language}_{config.paradigm}_{step}_finetuned.pth"))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                # forward pass
                loss = model(x, y, attn_blocksize=attn_blocksize, eos = 286 if (config.paradigm == "syl") else 288)
                # advance the dataset for the next batch
                x, y = train_loader.next_batch()
                # backward pass
                loss.backward()
        else: # just sync on the last step
            # forward pass
            loss = model(x, y, attn_blocksize=attn_blocksize, eos = 286 if (config.paradigm == "syl") else 288)
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()
        train_loss = loss.detach()
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # momentum warmup for Muon
    frac = min(step/300, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    if master_process:
        wandb.log(
            {'trainloss': train_loss},
            step=step)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

if master_process:
    if cli_args.pretraining:
        log = dict(model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, os.path.join(cli_args.out_path, f"{config.language}_{config.paradigm}_good.pth"))
    else:
        log = dict(model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, os.path.join(cli_args.out_path, f"{cli_args.task}_{config.language}_{config.paradigm}_{cli_args.cross_val_counter}_finetuned.pth"))

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()

# if master_process:
#     SENDER = "tiyliu@ucdavis.edu"
#     RECIPIENT = "tiyliu@ucdavis.edu"
#     SUBJECT = f"Training completed for {config.language}_{config.paradigm}"
#     BODY = "Training done."
#     APP_PASSWORD = email_password  # 16-character app password (no spaces when using)

#     send_email_gmail(SENDER, RECIPIENT, SUBJECT, BODY, APP_PASSWORD)
