import os
import sys
import json
import math
import time
import pickle
import random
import platform
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import logging
from logging.handlers import RotatingFileHandler
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model import Kronos, KronosTokenizer, KronosPredictor
from config_loader import CustomFinetuneConfig


def _validate_split_ratios(train_ratio, val_ratio, test_ratio):
    values = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }
    for name, value in values.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    total = sum(values.values())
    if total <= 0.0 or total > 1.0 + 1e-9:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must be in (0, 1], got "
            f"{total:.6f} from {values}"
        )
    return values


class CustomKlineDataset(Dataset):
    """Chronological train/val/test windows for next-token predictor training.

    A window contains ``lookback_window + predict_window`` bars.  The next-token
    targets therefore cover bars 1 .. window-1, and the forecast bars
    ``[lookback_window, lookback_window + predict_window)`` are exactly the last
    ``predict_window`` next-token targets.  No extra bar is appended.

    Split boundaries never move: training labels end before ``train_end``;
    validation windows may use earlier training history as context but their
    labels lie entirely in ``[train_end, val_end)``; test labels lie entirely in
    ``[val_end, n)``.
    """
    
    def __init__(self, data_path, data_type='train', lookback_window=90, predict_window=10, 
                 clip=5.0, seed=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        self.data_path = data_path
        self.data_type = data_type
        if data_type not in ('train', 'val', 'test'):
            raise ValueError(f"data_type must be train/val/test, got {data_type!r}")
        if int(lookback_window) < 1 or int(predict_window) < 1:
            raise ValueError("lookback_window and predict_window must both be >= 1")
        self.lookback_window = int(lookback_window)
        self.predict_window = int(predict_window)
        # No '+1': a lookback+predict window yields exactly `predict` next-token
        # forecasts, aligned with bars [lookback, lookback+predict).
        self.window = self.lookback_window + self.predict_window
        self.clip = clip
        self.seed = seed
        ratios = _validate_split_ratios(train_ratio, val_ratio, test_ratio)
        self.train_ratio = ratios["train_ratio"]
        self.val_ratio = ratios["val_ratio"]
        self.test_ratio = ratios["test_ratio"]
        
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        self.py_rng = random.Random(seed)
        self.current_epoch = 0
        self._epoch_order_cache = {}
        
        self._load_and_preprocess_data()
        self._split_data_by_time()
        self._build_indices()
            
        print(f"[{data_type.upper()}] Data length: {len(self.data)}, Available samples: {self.n_samples}")
    
    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)
        
        self.timestamps = df['timestamps'].copy()
        
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month
        
        self.data = df[self.feature_list + self.time_feature_list].copy()
        
        if self.data.isnull().any().any():
            print("Warning: Missing values found in data, performing forward fill")
            self.data = self.data.ffill()
        
        print(f"Original data time range: {self.timestamps.min()} to {self.timestamps.max()}")
        print(f"Original data total length: {len(df)} records")
    
    def _split_data_by_time(self):
        total_length = len(self.data)
        self.train_end = int(total_length * self.train_ratio)
        self.val_end = int(total_length * (self.train_ratio + self.val_ratio))
        if not (0 < self.train_end <= self.val_end <= total_length):
            raise ValueError(
                "Invalid chronological split: "
                f"n={total_length}, train_end={self.train_end}, val_end={self.val_end}, "
                f"ratios=({self.train_ratio}, {self.val_ratio}, {self.test_ratio})"
            )
        if self.data_type == 'train':
            lo, hi = 0, self.train_end
            print(f"[TRAIN] First {self.train_end} time points ({self.train_ratio:.3f})")
        elif self.data_type == 'val':
            lo, hi = self.train_end, self.val_end
            print(f"[VAL] Time points {self.train_end + 1} to {self.val_end} ({self.val_ratio:.3f})")
        else:
            lo, hi = self.val_end, total_length
            print(f"[TEST] Time points {self.val_end + 1} to {total_length}")
        print(f"[{self.data_type.upper()}] Split time range: {self.timestamps.iloc[lo]} to {self.timestamps.iloc[hi - 1]}")
    
    def _build_indices(self):
        """Build legal window starts without crossing a chronological boundary."""
        L = self.window
        n = len(self.data)
        if self.data_type == 'train':
            start_min = 0
            start_max = self.train_end - L
            boundary = self.train_end
        elif self.data_type == 'val':
            # Context may reach back into train, but labels must start at/after
            # train_end and end before val_end.
            start_min = max(0, self.train_end - self.lookback_window)
            start_max = self.val_end - L
            boundary = self.val_end
        else:
            start_min = max(0, self.val_end - self.lookback_window)
            start_max = n - L
            boundary = n
        
        if start_min > start_max:
            raise ValueError(
                f"Insufficient {self.data_type} data for lookback={self.lookback_window}, "
                f"predict={self.predict_window}: split_end={boundary}, n={n}, "
                f"legal window starts [{start_min}, {start_max}]"
            )
        self.start_indices = np.arange(start_min, start_max + 1, dtype=np.int64)
        if len(self.start_indices) == 0:
            raise ValueError(
                f"No legal {self.data_type} windows: lookback={self.lookback_window}, "
                f"predict={self.predict_window}, split_end={boundary}, n={n}"
            )
        self.n_samples = int(len(self.start_indices))
    
    def set_epoch_seed(self, epoch):
        self.current_epoch = int(epoch)
        self.py_rng.seed(self.seed + self.current_epoch)
    
    def _epoch_order(self, epoch):
        order = self._epoch_order_cache.get(epoch)
        if order is None:
            rng = random.Random(self.seed + 1000003 * int(epoch))
            order = list(range(self.n_samples))
            rng.shuffle(order)
            self._epoch_order_cache[epoch] = order
        return order
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        n = self.n_samples
        if n <= 0:
            raise ValueError("Data length insufficient to create samples")
        if idx < 0 or idx >= n:
            raise IndexError(f"index {idx} out of range for {self.data_type} dataset of size {n}")
        
        if self.data_type == 'train':
            start_idx = int(self.start_indices[self._epoch_order(self.current_epoch)[idx]])
        else:
            start_idx = int(self.start_indices[idx])
        end_idx = start_idx + self.window
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        x = window_data[self.feature_list].values.astype(np.float32)
        x_stamp = window_data[self.time_feature_list].values.astype(np.float32)
        
        # Training/inference normalization must never see forecast bars.
        past_x = x[:self.lookback_window]
        x_mean, x_std = np.mean(past_x, axis=0), np.std(past_x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)
        
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        
        return x_tensor, x_stamp_tensor


# ---------------------------------------------------------------------------
# Training helpers (pure functions are kept importable for regression tests)
# ---------------------------------------------------------------------------
def build_predict_only_loss_mask(batch_size, target_len, lookback_window,
                                 predict_window, device=None, dtype=torch.float32):
    """Return the padding mask expected by ``DualHead.compute_loss``.

    ``0`` means "valid target", ``1`` means "ignore" (the historical
    ``padding_mask`` semantics).  Token target ``j`` is bar ``j + 1``.  With a
    window of ``lookback + predict`` bars, the last ``predict`` targets are
    exactly bars ``[lookback, lookback + predict)``.  That is the forecast
    interval, with no extra future bar and no missing first forecast bar.
    """
    target_len = int(target_len)
    lookback_window = int(lookback_window)
    predict_window = int(predict_window)
    expected = lookback_window + predict_window - 1
    if target_len != expected:
        raise ValueError(
            "target_len does not match an aligned lookback+predict window: "
            f"expected {expected}, got {target_len}"
        )
    mask = torch.ones((int(batch_size), target_len), device=device, dtype=dtype)
    start = lookback_window - 1
    mask[:, start:start + predict_window] = 0.0
    return mask


def count_valid_targets(loss_mask):
    if loss_mask is None:
        return None
    return int((loss_mask == 0).sum().item())


def optimizer_steps_per_epoch(num_batches, accumulation_steps):
    accum = max(1, int(accumulation_steps))
    return int(math.ceil(int(num_batches) / accum))


def accumulation_group_size(batch_idx, num_batches, accumulation_steps):
    """Number of micro-batches in the accumulation group containing batch_idx.

    The final group may be partial; callers divide each micro-batch loss by this
    value so the accumulated gradient is still an average, not a smaller sum.
    """
    accum = max(1, int(accumulation_steps))
    num_batches = int(num_batches)
    if batch_idx < 0 or batch_idx >= num_batches:
        raise IndexError(f"batch_idx {batch_idx} outside [0, {num_batches})")
    group_start = (batch_idx // accum) * accum
    return int(min(accum, num_batches - group_start))


def _param_grad_norm(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def _model_forward_loss(model, token_in, token_out, stamps, loss_mask,
                        teacher_forcing=True):
    """Forward/next-token loss with deterministic s1 teacher forcing.

    The official ``Kronos.forward`` conditions s2 on a *sampled* s1 token when
    ``use_teacher_forcing=False``.  For a conditional s2 likelihood (and for
    deterministic validation) we condition on the actual s1 target instead.
    """
    logits = model(
        token_in[0], token_in[1], stamps,
        use_teacher_forcing=bool(teacher_forcing),
        s1_targets=token_out[0] if teacher_forcing else None,
    )
    core = model.module if hasattr(model, "module") else model
    loss, s1_loss, s2_loss = core.head.compute_loss(
        logits[0], logits[1], token_out[0], token_out[1],
        padding_mask=loss_mask,
    )
    return loss, s1_loss, s2_loss


def _git_commit(path):
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return None


def _jsonable_config(config):
    try:
        return json.loads(json.dumps(config.loader.config, default=str))
    except Exception:
        return {}


def collect_provenance(config, train_dataset=None, val_dataset=None, extra=None):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    vendored = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    provenance = {
        "schema": "kronos-ewz-training-provenance/1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_path": os.path.abspath(config.loader.config_path),
        "config": _jsonable_config(config),
        "experiment_name": getattr(config, "experiment_name", None),
        "exp_name": getattr(config, "exp_name", None),
        "seed": getattr(config, "seed", None),
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "project_git_commit": _git_commit(root),
        "vendored_git_commit": _git_commit(vendored),
        "loss_mask_predict_only": bool(getattr(config, "loss_mask_predict_only", False)),
        "teacher_forcing": bool(getattr(config, "teacher_forcing", True)),
        "accumulation_steps": int(getattr(config, "accumulation_steps", 1)),
        "validation_weighting": getattr(config, "validation_weighting", "uniform"),
        "early_stopping_patience": int(getattr(config, "early_stopping_patience", 0) or 0),
    }
    if train_dataset is not None:
        provenance["train_split"] = {
            "n_rows": int(len(train_dataset.data)),
            "train_end": int(train_dataset.train_end),
            "val_end": int(train_dataset.val_end),
            "n_windows": int(train_dataset.n_samples),
            "window": int(train_dataset.window),
            "start_index_first": int(train_dataset.start_indices[0]),
            "start_index_last": int(train_dataset.start_indices[-1]),
        }
    if val_dataset is not None:
        provenance["val_split"] = {
            "n_rows": int(len(val_dataset.data)),
            "train_end": int(val_dataset.train_end),
            "val_end": int(val_dataset.val_end),
            "n_windows": int(val_dataset.n_samples),
            "start_index_first": int(val_dataset.start_indices[0]),
            "start_index_last": int(val_dataset.start_indices[-1]),
        }
    if extra:
        provenance.update(extra)
    return provenance


def _write_json_atomic(path, payload):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


def ensure_run_path_available(base_save_path, basemodel_save_path, overwrite=False):
    """Fail instead of silently overwriting an existing experiment."""
    if overwrite:
        return
    existing = []
    for path in (base_save_path, basemodel_save_path):
        if path and os.path.isdir(path) and os.listdir(path):
            existing.append(path)
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing experiment run: "
            + ", ".join(existing)
            + ". Use a new exp_name/output path, or set overwrite_existing: true "
              "(or --overwrite-existing) explicitly."
        )


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    
    logger.info(f"=== Basemodel Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")
    
    train_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type='train',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    val_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type='val',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler
    )
    
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def train_model(model, tokenizer, device, config, save_dir, logger):
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; increase data size or reduce batch_size")
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty; check split ratios/window size")
    
    accumulation_steps = max(1, int(getattr(config, 'accumulation_steps', 1)))
    steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), accumulation_steps)
    weighting = str(getattr(config, 'validation_weighting', 'uniform')).lower()
    if weighting not in ('uniform', 'tokens'):
        raise ValueError("validation_weighting must be 'uniform' or 'tokens'")
    predict_only = bool(getattr(config, 'loss_mask_predict_only', False))
    teacher_forcing = bool(getattr(config, 'teacher_forcing', True))
    patience = int(getattr(config, 'early_stopping_patience', 0) or 0)
    min_delta = float(getattr(config, 'early_stopping_min_delta', 0.0) or 0.0)
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    core_model = model.module if use_ddp else model
    
    # The tokenizer is a frozen encoder throughout predictor training.
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=config.basemodel_epochs,
        pct_start=0.03,
        div_factor=10
    )
    
    os.makedirs(save_dir, exist_ok=True)
    provenance = collect_provenance(config, train_dataset, val_dataset, extra={
        "phase": "start",
        "device": str(device),
        "steps_per_epoch": int(steps_per_epoch),
        "micro_batches_per_epoch": int(len(train_loader)),
        "optimizer_steps_per_epoch": int(steps_per_epoch),
        "effective_batch_size": int(config.batch_size * accumulation_steps * world_size),
        "tokenizer_frozen": True,
    })
    _write_json_atomic(os.path.join(save_dir, "provenance.json"), provenance)
    
    best_val_loss = float('inf')
    best_epoch = None
    batch_idx_global = 0
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()
        tokenizer.eval()
        
        train_dataset.set_epoch_seed(epoch)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_train_loss = 0.0
        train_weight = 0.0
        train_micro_batches = 0
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            stamps = batch_x_stamp[:, :-1, :]
            
            loss_mask = None
            if predict_only:
                loss_mask = build_predict_only_loss_mask(
                    token_out[0].size(0), token_out[0].size(1),
                    config.lookback_window, config.predict_window,
                    device=token_out[0].device,
                )
            
            loss, s1_loss, s2_loss = _model_forward_loss(
                model, token_in, token_out, stamps, loss_mask,
                teacher_forcing=teacher_forcing,
            )
            
            group_size = accumulation_group_size(batch_idx, len(train_loader), accumulation_steps)
            (loss / group_size).backward()
            
            should_step = ((batch_idx + 1) % accumulation_steps == 0
                           or batch_idx == len(train_loader) - 1)
            if should_step:
                torch.nn.utils.clip_grad_norm_(core_model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            
            weight = count_valid_targets(loss_mask)
            if weight is None:
                weight = int(token_out[0].numel())
            epoch_train_loss += loss.item() * weight
            train_weight += weight
            train_micro_batches += 1
            
            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_msg = (f"[Epoch {epoch+1}/{config.basemodel_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {loss.item():.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)
            
            batch_idx_global += 1
        
        model.eval()
        tokenizer.eval()
        # No sampling occurs below (teacher forcing), but re-seeding makes the
        # validation pass deterministic even if a future change adds sampling.
        set_global_seed(config.seed)
        val_loss = 0.0
        val_weight = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                stamps = batch_x_stamp[:, :-1, :]
                
                # Validation context may reach into the training period.
                # Score forecast labels only, including when training uses full
                # next-token loss, so training-context labels cannot enter the
                # checkpoint selection criterion.
                loss_mask = build_predict_only_loss_mask(
                    token_out[0].size(0), token_out[0].size(1),
                    config.lookback_window, config.predict_window,
                    device=token_out[0].device,
                )
                loss, _, _ = _model_forward_loss(
                    model, token_in, token_out, stamps, loss_mask,
                    teacher_forcing=teacher_forcing,
                )
                
                weight = count_valid_targets(loss_mask)
                if weight is None:
                    weight = int(token_out[0].numel())
                if weighting == 'uniform':
                    weight = 1
                val_loss += loss.item() * weight
                val_weight += weight
                val_batches += 1
        
        if use_ddp:
            tensor_sum = torch.tensor(
                [epoch_train_loss, train_weight, val_loss, val_weight],
                dtype=torch.float64, device=device,
            )
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            epoch_train_loss_all = tensor_sum[0].item()
            train_weight_all = tensor_sum[1].item()
            val_loss_all = tensor_sum[2].item()
            val_weight_all = tensor_sum[3].item()
            avg_train_loss = (epoch_train_loss_all / train_weight_all) if train_weight_all > 0 else 0.0
            avg_val_loss = (val_loss_all / val_weight_all) if val_weight_all > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_weight if train_weight > 0 else 0.0
            avg_val_loss = val_loss / val_weight if val_weight > 0 else 0.0
        
        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
                       f"Training Loss: {avg_train_loss:.4f}\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Optimizer Steps: {optimizer_steps}/{steps_per_epoch}\n"
                       f"Epoch Time: {epoch_time:.2f} seconds\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)
        
        improved = avg_val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                core_model.save_pretrained(model_save_path)
                _write_json_atomic(os.path.join(model_save_path, "training_metadata.json"), {
                    "schema": "kronos-ewz-checkpoint-metadata/1",
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "seed": config.seed,
                    "experiment_name": getattr(config, "experiment_name", None),
                    "exp_name": getattr(config, "exp_name", None),
                    "loss_mask_predict_only": predict_only,
                    "teacher_forcing": teacher_forcing,
                    "accumulation_steps": accumulation_steps,
                    "validation_weighting": weighting,
                    "config_path": os.path.abspath(config.loader.config_path),
                })
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)
        else:
            epochs_without_improvement += 1
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "optimizer_steps": optimizer_steps,
            "micro_batches": train_micro_batches,
            "epoch_seconds": epoch_time,
        })
        _write_json_atomic(os.path.join(save_dir, "training_history.json"), history)
        
        if patience > 0 and epochs_without_improvement >= patience:
            stop_msg = (f"Early stopping after {epoch+1} epochs "
                        f"({epochs_without_improvement} without improvement > {min_delta:g}); "
                        f"best epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
            logger.info(stop_msg)
            if rank == 0:
                print(stop_msg)
            break
    
    provenance.update({
        "phase": "finished",
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    if rank == 0:
        _write_json_atomic(os.path.join(save_dir, "provenance.json"), provenance)
    
    return best_val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kronos Basemodel Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--overwrite-existing', action='store_true',
                       help='Explicitly allow writing into an existing experiment path')
    args = parser.parse_args()
    
    config = CustomFinetuneConfig(args.config)
    if args.overwrite_existing:
        config.overwrite_existing = True
    
    if config.use_cuda and torch.cuda.is_available():
        device_id = int(getattr(config, 'device_id', 0))
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"device_id={device_id} but only {torch.cuda.device_count()} CUDA device(s) available")
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
        if config.use_cuda:
            print("Warning: use_cuda=true but CUDA is unavailable; falling back to CPU")
    print(f"Using device: {device}")
    
    ensure_run_path_available(
        config.base_save_path, config.basemodel_save_path,
        overwrite=bool(getattr(config, 'overwrite_existing', False)),
    )
    os.makedirs(config.basemodel_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    
    set_global_seed(config.seed)
    
    logger.info("Loading pretrained model or random initialization...")
    print("Loading pretrained model or random initialization...")
    if getattr(config, 'pre_trained_tokenizer', True):
        tokenizer = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    else:
        print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture for training")
        cfg_path_tok = os.path.join(config.pretrained_tokenizer_path if hasattr(config, 'pretrained_tokenizer_path') else config.finetuned_tokenizer_path, 'config.json')
        with open(cfg_path_tok, 'r') as f:
            arch_t = json.load(f)
        tokenizer = KronosTokenizer(
            d_in=arch_t.get('d_in', 6),
            d_model=arch_t.get('d_model', 256),
            n_heads=arch_t.get('n_heads', 4),
            ff_dim=arch_t.get('ff_dim', 512),
            n_enc_layers=arch_t.get('n_enc_layers', 4),
            n_dec_layers=arch_t.get('n_dec_layers', 4),
            ffn_dropout_p=arch_t.get('ffn_dropout_p', 0.0),
            attn_dropout_p=arch_t.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch_t.get('resid_dropout_p', 0.0),
            s1_bits=arch_t.get('s1_bits', 10),
            s2_bits=arch_t.get('s2_bits', 10),
            beta=arch_t.get('beta', 0.05),
            gamma0=arch_t.get('gamma0', 1.0),
            gamma=arch_t.get('gamma', 1.1),
            zeta=arch_t.get('zeta', 0.05),
            group_size=arch_t.get('group_size', 4)
        )

    if getattr(config, 'pre_trained_predictor', True):
        model = Kronos.from_pretrained(config.pretrained_predictor_path)
    else:
        print("pre_trained_predictor=False, randomly initializing Predictor architecture for training")
        cfg_path = os.path.join(config.pretrained_predictor_path, 'config.json')
        with open(cfg_path, 'r') as f:
            arch = json.load(f)
        model = Kronos(
            s1_bits=arch.get('s1_bits', 10),
            s2_bits=arch.get('s2_bits', 10),
            n_layers=arch.get('n_layers', 12),
            d_model=arch.get('d_model', 832),
            n_heads=arch.get('n_heads', 16),
            ff_dim=arch.get('ff_dim', 2048),
            ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
            attn_dropout_p=arch.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch.get('resid_dropout_p', 0.2),
            token_dropout_p=arch.get('token_dropout_p', 0.0),
            learn_te=arch.get('learn_te', True)
        )
    
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    model = model.to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_size:,}")
    print(f"Model parameters: {model_size:,}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.predictor_learning_rate}")
    logger.info(f"Training epochs: {config.basemodel_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Tokenizer path: {config.finetuned_tokenizer_path}")
    logger.info(f"Pretrained model path: {config.pretrained_predictor_path}")
    
    logger.info("Starting fine-tuning training...")
    print("Starting fine-tuning training...")
    best_val_loss = train_model(model, tokenizer, device, config, config.basemodel_save_path, logger)
    
    final_msg = f"Training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.basemodel_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
