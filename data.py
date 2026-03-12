import torch

from prepare import (
    _document_batches as _raw_document_batches,
    MAX_SEQ_LEN,
)


def sample_documents(n=10):
    docs = []
    for batch, _ in _raw_document_batches("train"):
        docs.extend(batch[:n - len(docs)])
        if len(docs) >= n:
            break
    return docs[:n]


def filter_document(text):
    if len(text) < 100:
        return False
    return True


def process_document(text):
    if len(text) > 6000:
        text = text[:6000]
    return text


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    raw_batches = _raw_document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(raw_batches)
        filtered = [text[:6000] for text in doc_batch if len(text) >= 100]
        if filtered:
            token_lists = tokenizer.encode(filtered, prepend=bos_token)
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
                shortest_idx = 0
                shortest_len = 999999
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                    if doc_len < shortest_len:
                        shortest_idx = i
                        shortest_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch
