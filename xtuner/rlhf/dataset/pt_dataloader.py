# TODO https://github.com/InternLM/xtuner/blob/main/xtuner/dataset/intern_repo.py

import os
import time
import itertools
import operator
import numpy as np
import torch
from torch.utils.data import ConcatDataset
import threading
from typing import Dict
from tqdm import tqdm
from pathlib import Path
import mmap
import json
from copy import deepcopy


DATASET_TYPE_IDS_MAP = {"en": 0, "cn": 1, "code": 2, "ja": 3, "ar": 4, "kaoshi": 5}
def get_dataset_type_id(path):
    import re

    match_idxes = []
    for key, idx in DATASET_TYPE_IDS_MAP.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {DATASET_TYPE_IDS_MAP}"
    return match_idxes[0]


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "tokens": List[int],
    }
    ```

    Note that only the "tokens" key is used.
    """

    def __init__(
        self,
        path: str,
        dataset_type_id: int = 0,  # Used to indicate what type of dataset the current dataset is, such as en/cn/code,
        # the specific mapping can be found from data.utils:DATASET_TYPE_IDS_MAP
    ):
        self.path = path

        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.cache = Path(f"{resolved_path}.meta")
        # only build the cache in on the primary worker to prevent overloading nfs
        assert os.path.exists(self.cache), f"The cache file:{self.cache} is not found for file:{self.path}"
        try:
            with open(self.cache, "rb") as f:
                meta = np.load(f)
        except Exception as e:
            print(f"Cannot load file {resolved_path}...")
            raise e
        self.offsets = meta[:, 0]
        self.lengths = meta[:, -1]
        self.type_id = dataset_type_id

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            with open(self.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith(".gz") or self.path.endswith(".bz") or self.path.endswith(".bz2"):
                    raise NotImplementedError(
                        "Compressed files are not supported because .seek() would require "
                        "rereading the entire file, making performance too slow."
                    )
        return self.threadlocal.handles[-1]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode("utf-8")
        try:
            item = json.loads(item)
            item["length"] = len(item["tokens"])  # add a length info
            item["type_id"] = self.type_id
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(
                    f"Error while loading JSONL line in file {self.path} at byte "
                    f"{position}. Contents of line:\n{item}\n{err}"
                ),
            )
        return item

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number if the number of documents
        # is not perfectly divisible by the data_subshard_count
        return len(self.offsets)

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class ConcatDatasetWrapper(ConcatDataset):
    """A wrapper of concat dataset for packing."""
    def __init__(self, datasets):
        super().__init__(datasets=datasets)
        self._lengths = np.concatenate([ds.lengths for ds in datasets], axis=-1)

    @property
    def lengths(self):
        return self._lengths
    
    def get_dataset_name(self):
        return "ConcatDatasetWrapper"


class PackedDataset(torch.utils.data.Dataset):
    """
    The class PackedDataset takes in a dataset and aggregates samples of different
    lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        packed_length: The length of each packed sample. Default is 4096.
    """

    def __init__(
        self,
        dataset,
        packed_length: int = 4096,
    ):
        assert hasattr(dataset, "lengths")
        assert len(getattr(dataset, "lengths")) == len(
            dataset
        ), "The dataset must have lengths attribute and have the same length as the dataset"
        self.dataset = dataset
        self.lengths = getattr(self.dataset, "lengths")
        self.packed_length = packed_length
        # Force a seed to be fixed to prevent problems caused by the seed not being restored when restarting
        self.seed = 1024
        self.sample_indices, self.len_samples_shuffled, self.acm_len_samples = self.accu_sample_len(seed=self.seed)
        self.num_tokens = sum(self.lengths)

    def get_dataset_name(self):
        return self.dataset.get_dataset_name()

    def accu_sample_len(self, seed=None):
        """accumulative length of samples"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(self.seed - 1)

        sample_indices = np.arange(len(self.lengths))
        rng.shuffle(sample_indices)
        len_samples_shuffled = list(map(self.lengths.__getitem__, sample_indices))
        acm_len_samples = list(itertools.accumulate(len_samples_shuffled, operator.add))
        return sample_indices, len_samples_shuffled, acm_len_samples

    def __len__(self):
        # Line 405 of document_to_sequence.py in metaseq is directly spliced,
        # without additional consideration of sos or eos
        n_packs = self.num_tokens // self.packed_length
        return n_packs

    def cal_map(self, carriage_idx: int = 0):
        assert carriage_idx >= 0
        length_train = (carriage_idx + 1) * self.packed_length
        post_pos = np.searchsorted(self.acm_len_samples, length_train, side="left")
        return post_pos

    def mapping(self, pack_idx: int = 0):
        # pack_idx is zero-based
        pre_pos, pre_token_id = 0, 0
        if pack_idx > 0:
            pre_pos = self.cal_map(pack_idx - 1)
            pre_token_id = self.len_samples_shuffled[pre_pos] - (
                self.acm_len_samples[pre_pos] - (pack_idx) * self.packed_length
            )
            if pre_token_id == self.len_samples_shuffled[pre_pos]:
                pre_pos += 1
                pre_token_id = 0

        pos = self.cal_map(pack_idx)
        token_id = self.len_samples_shuffled[pos] - (self.acm_len_samples[pos] - (pack_idx + 1) * self.packed_length)
        return pre_pos, pre_token_id, pos, token_id

    def build_pack(self, pre_pos: int, pre_token_id: int, pos: int, token_id: int):
        pack, cu_seqlens, indexes, labels, type_ids = [], [0], [], [], []

        while pre_pos < pos:
            sample_idx = self.sample_indices[pre_pos]
            sample = self.dataset[sample_idx]
            chunk = sample["tokens"][pre_token_id:]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            _labels = list(_labels[1:]) + [-100]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            type_ids.extend([sample.get("type_id", 0)] * len(chunk))
            num_new_samples, tokens_left = divmod(len(chunk), self.packed_length)
            for _ in range(num_new_samples):
                cu_seqlens.append(cu_seqlens[-1] + self.packed_length)
                indexes.extend(list(range(self.packed_length)))
            if tokens_left > 0:
                cu_seqlens.append(cu_seqlens[-1] + tokens_left)
                indexes.extend(list(range(tokens_left)))
            pre_pos = pre_pos + 1
            pre_token_id = 0

        sample_idx = self.sample_indices[pos]
        sample = self.dataset[sample_idx]
        chunk = sample["tokens"][pre_token_id:token_id]  # fragement of a sample
        pack.extend(chunk)
        _labels = deepcopy(chunk)
        if token_id == len(sample["tokens"]):
            _labels = list(_labels[1:]) + [-100]
        else:
            if token_id > len(sample["tokens"]):
                print(f"token_id {token_id}, len of sample {len(sample['tokens'])}")
            _labels = list(_labels[1:]) + [sample["tokens"][token_id]]
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        type_ids.extend([sample.get("type_id", 0)] * len(chunk))
        num_new_samples, tokens_left = divmod(len(chunk), self.packed_length)
        for _ in range(num_new_samples):
            cu_seqlens.append(cu_seqlens[-1] + self.packed_length)
            indexes.extend(list(range(self.packed_length)))
        if tokens_left > 0:
            cu_seqlens.append(cu_seqlens[-1] + tokens_left)
            indexes.extend(list(range(tokens_left)))

        out = {"tokens": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels, "type_ids": type_ids}
        return out

    def __getitem__(self, item: int) -> Dict:
        """Given the index, it returns a dict as
        {
         'tokens': List[int],
         'cu_seqlens': List[int],
         'indexes': List[int], # denotes positional vector as 'tokens'
         'labels': List[int], # corresponds to 'tokens' and shifted yet, -100 means skipping prediction
        }
        """
        pos_before, token_id_before, pos_after, token_id_after = self.mapping(item)
        return self.build_pack(pos_before, token_id_before, pos_after, token_id_after)
    

def get_concat_packed_dataset(
    folder,
    packed_length=4096,
    show_progress=False,
):
    """
    Given a folder, combine all the .bin into a single large dataset.

    All .bin files are concatenated into a single dataset.

    Args:
        folder (str): Path to the folder containing the .bin files.
        packed_length (int): Length to pack samples to.
        show_progress (bool): Whether to show the progress bar.

    Returns:
        A packed dataset containing all the data from the .bin files.
    """

    assert os.path.exists(folder), f"{folder} does not exist."
    datasets = []

    time_json, time_pack = 0, 0

    for root, dirs, files in os.walk(folder, followlinks=True):
        dirs.sort()  # Let the folder need to be returned in a fixed order
        print(f"Reading {root}...")
        for fn in tqdm(sorted(files), total=len(files), leave=False, disable=not show_progress):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                ds_type_id = get_dataset_type_id(path=fp)

                s = time.time()
                ds = JsonlDataset(fp, dataset_type_id=ds_type_id)
                time_json += time.time() - s

                if len(ds) == 0:
                    continue
                datasets.append(ds)

    dataset = ConcatDatasetWrapper(datasets=datasets)

    s = time.time()
    dataset = PackedDataset(dataset, packed_length)
    time_pack += time.time() - s
    print(f"json time: {time_json:.3f}s pack time: {time_pack:.3f}s")
    # torch.distributed.barrier()
    print(
        f"In total, find `{len(datasets)}` datasets, {len(dataset)} samples," \
        f" total tokens {dataset.num_tokens}",
    )

    return dataset


def packed_collate_fn(batch, packed_length):
    """
    Collate function for packed input sequences.

    Args:
        batch (List[Dict]): List of dictionaries representing each sample in batch.
            Each dictionary contains "tokens", "labels", "type_ids", "cu_seqlens", and "indexes" keys.
        packed_length (int): The length of packed sequence.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing a dictionary of tensors with "input_ids",
            "cu_seqlens", "indexes", and "type_ids" keys, and the tensor of padded "labels".

    Raises:
        AssertionError: If the length of a sample is not equal to packed_length.
        AssertionError: If the shape of the padded "input_ids" tensor does not have the correct shape.
    """

    xs, ys, cu_seqlens, indexes, ts = [], [], [], [], []
    for b in batch:
        assert (
            len(b["tokens"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['tokens'])} and {packed_length})"
        assert (
            len(b["labels"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])} and {packed_length})"
        assert (
            len(b["type_ids"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['type_ids'])} and {packed_length})"

        tokens = [abs(w) for w in b["tokens"]]
        labels = [w if w > 0 else -100 for w in b["labels"]]

        xs.append(torch.LongTensor(tokens))
        # The labels have been shifted here, so they are aligned with the output corresponding to the token
        ys.append(torch.LongTensor(labels))
        ts.append(torch.LongTensor(b["type_ids"]))
        cu_seqlens.append(b["cu_seqlens"])
        indexes.append(torch.LongTensor(b["indexes"]))

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    indexes = torch.stack(indexes, dim=0)
    if len(set(map(len, cu_seqlens))) == 1:  # if has uniform length, then stack to save device transfer time
        cu_seqlens = torch.stack(cu_seqlens, dim=0)

    assert xs.shape[1] == packed_length, (xs.shape[1], packed_length)

    # return {"input_ids": xs, "cu_seqlens": cu_seqlens, "indexes": indexes, "type_ids": ts}, ys
    return {"input_ids": xs, "attention_mask":None, "cumulative_len": cu_seqlens, "position_ids": indexes, "labels": ys}

def get_pretrain_data(folder, batch_size, length=None, shuffle=False):
    import functools
    from torch.utils.data import DataLoader
    pretrain_dataset = get_concat_packed_dataset(
                folder=folder,
                packed_length=length,
            )
    pretrain_dataloader = DataLoader(
            shuffle=shuffle,
            dataset=pretrain_dataset,
            batch_size=batch_size,
            collate_fn=functools.partial(packed_collate_fn, packed_length=length),
        )

    pretrain_data_iterator = pretrain_dataloader.__iter__()
    return pretrain_data_iterator


if __name__ == "__main__":
    import functools
    from torch.utils.data import DataLoader
    pretrain_data_iterator = get_pretrain_data(folder="/fs-computility/llm/shared/zhaoqian/dataset/pretrain/1226-mix-v13-complete-watermark-pjx50/train",
                                                        # folder="/cpfs01/shared/public/public_hdd/llmit_new/ppo/dataset/pretrain/1226-mix-v13-complete-watermark-pjx50/train", 
                                                        batch_size=16,
                                                        length=32768,)

    pretrain_data = next(pretrain_data_iterator)

    import numpy as np
    import torch
    torch.set_printoptions(threshold=np.inf)

    print(len(pretrain_data))
    for k in pretrain_data.keys():
        if k != 'cumulative_len':
            print(k, pretrain_data[k].shape, pretrain_data[k][0])

    print(pretrain_data['cumulative_len'])
    print(pretrain_data.keys())
    
