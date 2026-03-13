import os
import torch
import pyarrow.parquet as pq

from prepare import (
    _document_batches as _raw_document_batches,
    MAX_SEQ_LEN,
    DATA_DIR,
    VAL_FILENAME,
)


def _interleaved_document_batches(split, tokenizer_batch_size=128):
    """Read from shards in round-robin order for better diversity."""
    parquet_paths = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]

    epoch = 1
    while True:
        # Open all shards and get row group iterators
        shard_iters = []
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            shard_iters.append((pf, 0))  # (file, current_row_group_idx)

        # Round-robin across shards, one row group at a time
        active = list(range(len(shard_iters)))
        while active:
            next_active = []
            for idx in active:
                pf, rg_idx = shard_iters[idx]
                if rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    shard_iters[idx] = (pf, rg_idx + 1)
                    next_active.append(idx)
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i + tokenizer_batch_size], epoch
            active = next_active
        epoch += 1


def _document_batches(split, tokenizer_batch_size=128):
    return _interleaved_document_batches(split, tokenizer_batch_size)


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch
