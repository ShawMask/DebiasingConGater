import os
import pickle
import argparse
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoConfig

from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import RTE, MRPC, CoLA, STSB, SST2, QNLI, QQP, MNLI

from typing import Union, Tuple, List, Optional, Any


class TextDS(Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.text = text
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int):
        tokenized_sample = self.tokenizer(
            self.text[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        token_ids_sample = tokenized_sample['input_ids']
        # When Roberta is used it Tokenizer does not have token_type_ids
        try:
            token_type_ids_sample = tokenized_sample['token_type_ids']
        except:
            token_type_ids_sample = torch.zeros_like(token_ids_sample)
        attention_masks_sample = tokenized_sample['attention_mask']
        return token_ids_sample, token_type_ids_sample, attention_masks_sample


def multiprocess_tokenization(text_list, tokenizer, max_length, num_workers=16):
    ds = TextDS(text_list, tokenizer, max_length)
    _loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=num_workers, drop_last=False)
    token_ids = []
    token_type_ids = []
    attention_masks = []
    for tokenized_batch, token_type_ids_batch, attention_masks_batch in _loader:
        token_ids.append(tokenized_batch)
        token_type_ids.append(token_type_ids_batch)
        attention_masks.append(attention_masks_batch)

    token_ids = torch.cat(token_ids, dim=0).squeeze(1)
    token_type_ids = torch.cat(token_type_ids, dim=0).squeeze(1)
    attention_masks = torch.cat(attention_masks, dim=0).squeeze(1)

    return token_ids, token_type_ids, attention_masks


def read_label_file(filepath):
    with open(filepath) as f:
        data = f.read()
        return {v:k for k,v in enumerate([l for l in data.split("\n") if len(l)>0])}


def get_data_loader(
    task_key: str,
    protected_key: Union[str, list, tuple],
    text_key: str,
    tokenizer: Tokenizer,
    data_path: Union[str, os.PathLike],
    labels_task_path: Union[str, os.PathLike],
    labels_prot_path: Optional[Union[str, os.PathLike, list, tuple]] = None,
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
    debug: bool = False,
    only_load_dataset: bool = False,
    external_data: dict = None
):

    def batch_fn(batch):
        input_ids, token_type_ids, attention_masks, labels_task = [torch.stack(l) for l in zip(*batch)]
        x = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks
        }
        return x, labels_task


    def batch_fn_prot(batch):
        # b = input_ids, token_type_ids, attention_masks, labels_task, labels_prot1, labels_prot2, ...
        b = [torch.stack(l) for l in zip(*batch)]
        x = {
            "input_ids": b[0],
            "token_type_ids": b[1],
            "attention_mask": b[2]
        }
        return x, *b[3:]


    if isinstance(protected_key, str):
        protected_key = [protected_key]
    if isinstance(labels_prot_path, str):
        labels_prot_path = [labels_prot_path]

    with open(data_path, 'rb') as file:
        data_dicts = pickle.load(file)

    if debug:
        cutoff = min(int(batch_size*10), len(data_dicts))
        data_dicts = data_dicts[:cutoff]
    keys = [task_key, *protected_key, text_key]
    x = [[d[k] for k in keys] for d in data_dicts]

    data = dict(zip(keys, zip(*x)))
    data[text_key] = list(data[text_key])
    for i, text in enumerate(data[text_key]):
        data[text_key][i] = text.replace("  ", " ")
        data[text_key][i] = text.lower()
    if only_load_dataset:
        return data
    else:
        if external_data:
            data = external_data
        input_ids, token_type_ids, attention_masks = multiprocess_tokenization(data[text_key], tokenizer, max_length)


        labels_task = read_label_file(labels_task_path)
        labels_task = torch.tensor([labels_task[str(t)] for t in data[task_key]], dtype=torch.long)

        tds = [
            input_ids,
            token_type_ids,
            attention_masks,
            labels_task
        ]

        if labels_prot_path is not None:
            for k, f in zip(protected_key, labels_prot_path):
                labels_prot = read_label_file(f)
                tds.append(torch.tensor([labels_prot[str(t)] for t in data[k]], dtype=torch.long))
                collate_fn = batch_fn_prot
        else:
            collate_fn = batch_fn

        _dataset = TensorDataset(*tds)

        return DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=collate_fn)


def get_data_loader_triplets(
    task_key: str,
    protected_key: Union[str, list, tuple],
    text_key: str,
    tokenizer: Tokenizer,
    data_path: Union[str, os.PathLike],
    labels_task_path: Union[str, os.PathLike],
    labels_prot_path: Union[str, os.PathLike, list, tuple],
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
    debug: bool = False
):

    def batch_fn_triplets(batch):

        def make_input_dict(v):
            return {
                "input_ids": v[0],
                "token_type_ids": v[1],
                "attention_mask": v[2]
            }

        # b = input_ids, token_type_ids, attention_masks, labels_task, labels_prot1, labels_prot2, ...
        b = [torch.stack(l) for l in zip(*batch)]
        anchor = make_input_dict(b[:3])
        negative = make_input_dict(b[3:6])
        positive = make_input_dict(b[6:9])

        return anchor, negative, positive, *b[9:]

    if isinstance(protected_key, str):
        protected_key = [protected_key]
    if isinstance(labels_prot_path, str):
        labels_prot_path = [labels_prot_path]

    with open(data_path, 'rb') as file:
        data_dicts = pickle.load(file)


    if debug:
        cutoff = min(int(batch_size*10), len(data_dicts))
        data_dicts = data_dicts[:cutoff]

    triplet_key_pos = "input_other_pv"
    triplet_key_neg = "input_other_tv"
    keys = [task_key, *protected_key, text_key] + [triplet_key_pos, triplet_key_neg]
    x = [[d[k] for k in keys] for d in data_dicts]

    data = dict(zip(keys, zip(*x)))

    tds = []
    for col in [text_key, triplet_key_neg, triplet_key_pos]:
        if col == triplet_key_pos:
            texts, weights = zip(*data[col])
        else:
            texts = data[col]
        input_ids, token_type_ids, attention_masks = multiprocess_tokenization(texts, tokenizer, max_length)
        tds.extend([input_ids, token_type_ids, attention_masks])
    tds.append(torch.tensor(weights))

    labels_task = read_label_file(labels_task_path)
    labels_task = torch.tensor([labels_task[str(t)] for t in data[task_key]], dtype=torch.long)
    tds.append(labels_task)

    for k, f in zip(protected_key, labels_prot_path):
        labels_prot = read_label_file(f)
        labels_prot = torch.tensor([labels_prot[str(t)] for t in data[k]], dtype=torch.long)
        tds.append(labels_prot)

    _dataset = TensorDataset(*tds)

    return DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=batch_fn_triplets)


def get_num_labels(label_file: Union[str, os.PathLike]) -> int:
    num_labels = len(read_label_file(label_file))
    return 1 if num_labels==2 else num_labels


def get_max_length(data_paths: List[Union[str, os.PathLike]], text_key: str) -> int:

    data_dicts = []
    for p in data_paths:
        with open(p, 'rb') as file:
            data_dicts.extend(pickle.load(file))

    texts = [d[text_key] for d in data_dicts]
    return max([len(x) for x in texts])


def get_class_weights(
    data_path: Union[str, os.PathLike],
    label_file_path: Union[str, os.PathLike, list, tuple],
    label_key: Union[str, list, tuple]
) -> list:
    if isinstance(label_file_path, str) or isinstance(label_file_path, os.PathLike):
        assert isinstance(label_key, str), "if only one label_file_path is provided label_key needs to by string"
        label_file_path = [label_file_path]
        label_key = [label_key]
    else:
        assert isinstance(label_key, (list, tuple)), "label_key needs to have same length as label_file_path"

    with open(data_path, 'rb') as f:
        data_dicts = pickle.load(f)

    res = []
    for p, k in zip(label_file_path, label_key):
        label_mapping = read_label_file(p)
        labels = [label_mapping[d[k]] for d in data_dicts]
        class_weights = compute_class_weight(class_weight='balanced', classes=list(set(labels)), y=labels)
        res.append(class_weights.tolist())
    return res


def get_data(
    args_train: argparse.Namespace,
    use_all_attr: bool = False,
    attr_idx_prot: Optional[int] = None,
    compute_class_weights: bool = False,
    device: Union[str, torch.device] = torch.device("cpu"),
    triplets: bool = False,
    debug: bool = False
) -> Tuple[DataLoader, DataLoader, int, int, list, str]:
    assert type(args_train.labels_protected_path) == type(args_train.protected_key), \
        "labels_protected_path and protected_needs to have same type"
    num_labels = get_num_labels(args_train.labels_task_path)

    if isinstance(args_train.labels_protected_path, (list, tuple)):
        if use_all_attr:
            labels_prot_path = args_train.labels_protected_path
        else:
            labels_prot_path = [args_train.labels_protected_path[attr_idx_prot]]
    else:
        labels_prot_path = [args_train.labels_protected_path]
    num_labels_protected_list = [get_num_labels(x) for x in labels_prot_path]

    if isinstance(args_train.protected_key, (list, tuple)):
        if use_all_attr:
            key_prot_list = args_train.protected_key
        else:
            key_prot_list = [args_train.protected_key[attr_idx_prot]]
    else:
        key_prot_list = [args_train.protected_key]
    len_key_prot = len(key_prot_list)

    cfg = AutoConfig.from_pretrained(args_train.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)

    length_check = [
        get_max_length([args_train.train_pkl, args_train.val_pkl], args_train.text_key),
        cfg.max_position_embeddings
    ]
    if args_train.tokenizer_max_length is not None:
        length_check.append(args_train.tokenizer_max_length)
    max_length = min(length_check)

    if compute_class_weights:
        num_classes_protected = get_class_weights(args_train.train_pkl, labels_prot_path, key_prot_list)
        protected_class_weights_list = [torch.tensor(n, device=device) for n in num_classes_protected]
    else:
        protected_class_weights_list = [None] * len_key_prot

    if triplets:
        if use_all_attr and len(key_prot_list)>1:
            triplet_pkl = args_train.train_triplet_all
        else:
            if isinstance(args_train.train_triplet, (list, tuple)):
                triplet_pkl = args_train.train_triplet[attr_idx_prot]
            else:
                triplet_pkl = args_train.train_triplet
        train_loader = get_data_loader_triplets(
            task_key = args_train.task_key,
            protected_key = key_prot_list,
            text_key = args_train.text_key,
            tokenizer = tokenizer,
            data_path = triplet_pkl,
            labels_task_path = args_train.labels_task_path,
            labels_prot_path = labels_prot_path,
            batch_size = args_train.batch_size,
            max_length = max_length,
            debug = debug
        )
    else:

        train_loader = get_data_loader(
            task_key = args_train.task_key,
            protected_key = key_prot_list,
            text_key = args_train.text_key,
            tokenizer = tokenizer,
            data_path = args_train.train_pkl,
            labels_task_path = args_train.labels_task_path,
            labels_prot_path = labels_prot_path,
            batch_size = args_train.batch_size,
            max_length = max_length,
            debug = debug
        )

    val_loader = get_data_loader(
        task_key = args_train.task_key,
        protected_key = key_prot_list,
        text_key = args_train.text_key,
        tokenizer = tokenizer,
        data_path = args_train.val_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = labels_prot_path,
        batch_size = args_train.batch_size,
        max_length = max_length,
        shuffle = False,
        debug = debug
    )
    test_loader = get_data_loader(
        task_key = args_train.task_key,
        protected_key = key_prot_list,
        text_key = args_train.text_key,
        tokenizer = tokenizer,
        data_path = args_train.test_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = labels_prot_path,
        batch_size = args_train.batch_size,
        max_length = max_length,
        shuffle = False,
        debug = debug
    )

    return train_loader, test_loader, val_loader, num_labels, num_labels_protected_list, key_prot_list, protected_class_weights_list


class GeluDataset(Dataset):

    def __init__(self, iterator, args_train):
        self.ds = args_train.ds
        if self.ds in ["rte", "mrpc", "qnli", "qqp", "mnli"]:
            self.label_idx = 0
        elif self.ds in ["cola", "stsb", "sst2"]:
            self.label_idx = 1
        self.max_length = args_train.tokenizer_max_length
        self.data = list(iterator)
        self.tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)

    def __getitem__(self, item):
        if self.ds == "cola":
            text = f"{self.data[item][2]}"
        elif self.ds in ["rte", "mrpc", "cola", "qnli", "qqp", "mnli"]:
            text = f"{self.data[item][1]} {self.data[item][2]}"
        elif self.ds in ["stsb"]:
            text = f"{self.data[item][2]} {self.data[item][3]}"
        elif self.ds in ["sst2"]:
            text = f"{self.data[item][0]}"
        else:
            raise NotImplementedError

        x = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        x = {k: v.squeeze(0) for k, v in x.items()}
        # print(torch.tensor(torch.round(torch.tensor(self.data[item][self.label_idx])), dtype=torch.long), self.data[item][self.label_idx])
        if self.ds in ["stsb"]:
            label = torch.tensor(torch.round(torch.tensor(self.data[item][self.label_idx])), dtype=torch.long)
        else:
            label = torch.tensor(self.data[item][self.label_idx])
        # print(label)
        return x, label

    def __len__(self):
        return len(self.data)


def get_glu_data(
    dataset: str,
    args_train: argparse.Namespace,
    num_workers: int = 15,
    debug: bool = False
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], int, int, int, list[Any]]:
    print(f"GLUE {dataset} Dataset is Loading")
    model_class = {"rte": RTE,  "mrpc": MRPC, "cola": CoLA,
                   "stsb": STSB, "sst2": SST2, "qnli": QNLI,
                   "qqp": QQP, "mnli": MNLI}
    iterator = model_class[dataset]()
    if dataset == "qqp":
        iterator = [iterator, iterator]

    print("Sample Sentence:")
    print(list(iterator[0])[1])
    num_word = 0
    counter = 0

    for i, text in enumerate(iterator[0]):
        if i == 0:
            if dataset in ["rte", "mrpc", "qnli", "qqp", "mnli"]:
                max_word = len(f"{text[1]} {text[2]}".split(" "))
            elif dataset in ["stsb"]:
                max_word = len(f"{text[2]} {text[3]}".split(" "))
            elif dataset in ["sst2"]:
                max_word = len(f"{text[0]}".split(" "))
            elif dataset in ["cola"]:
                max_word = len(f"{text[2]}".split(" "))
        else:
            if dataset in ["rte", "mrpc", "qnli", "qqp", "mnli"]:
                len_ = len(f"{text[1]} {text[2]}".split(" "))
            elif dataset in ["stsb"]:
                len_ = len(f"{text[2]} {text[3]}".split(" "))
            elif dataset in ["sst2"]:
                len_ = len(f"{text[0]}".split(" "))
            elif dataset in ["cola"]:
                len_ = len(f"{text[2]}".split(" "))
            if len_ > max_word:
                max_word = len_
        counter += 1
        if dataset in ["rte", "mrpc", "qnli", "qqp", "mnli"]:
            num_word += len(f"{text[1]} {text[2]}".split(" "))
        elif dataset in ["cola"]:
            num_word += len(f"{text[2]}".split(" "))
        elif dataset in ["stsb"]:
            num_word += len(f"{text[2]} {text[3]}".split(" "))
        elif dataset in ["sst2"]:
            num_word += len(f"{text[0]}".split(" "))

    print("Mean Number of Words:", num_word/counter)
    print("Max Number of Words:", max_word)
    if dataset in ["rte", "cola", "mrpc", "sst2", "qnli", "qqp"]:
        num_labels = 2
    elif dataset in ["stsb"]:
        num_labels = 6
    elif dataset in ["mnli"]:
        num_labels = 3
    num_labels_protected_list = 2
    key_prot_list = 2
    len_key_prot = 2

    train_dataset = GeluDataset(iterator[0], args_train)
    val_dataset = GeluDataset(iterator[1], args_train)

    train_loader = DataLoader(train_dataset, batch_size=args_train.batch_size,
                              shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args_train.batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers)

    if dataset in ["rte", "mrpc", "qqp", "mnli"]:
        test_loader = DataLoader(val_dataset, batch_size=args_train.batch_size)
    else:
        test_dataset = GeluDataset(iterator[2], args_train)
        test_loader = DataLoader(test_dataset, batch_size=args_train.batch_size)

    return train_loader, test_loader, val_loader, num_labels, num_labels_protected_list, key_prot_list, []


def custom_data_loader(
        args_train: argparse.Namespace,
        use_all_attr: bool = False,
        attr_idx_prot: Optional[int] = None,
        compute_class_weights: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
        triplets: bool = False,
        debug: bool = False,
        return_dataset = False,
        external_data = None):

    assert type(args_train.labels_protected_path) == type(args_train.protected_key), \
        "labels_protected_path and protected_needs to have same type"
    num_labels = get_num_labels(args_train.labels_task_path)

    if isinstance(args_train.labels_protected_path, (list, tuple)):
        if use_all_attr:
            labels_prot_path = args_train.labels_protected_path
        else:
            labels_prot_path = [args_train.labels_protected_path[attr_idx_prot]]
    else:
        labels_prot_path = [args_train.labels_protected_path]
    num_labels_protected_list = [get_num_labels(x) for x in labels_prot_path]

    if isinstance(args_train.protected_key, (list, tuple)):
        if use_all_attr:
            key_prot_list = args_train.protected_key
        else:
            key_prot_list = [args_train.protected_key[attr_idx_prot]]
    else:
        key_prot_list = [args_train.protected_key]
    len_key_prot = len(key_prot_list)

    cfg = AutoConfig.from_pretrained(args_train.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)

    length_check = [
        get_max_length([args_train.train_pkl, args_train.val_pkl], args_train.text_key),
        cfg.max_position_embeddings
    ]
    if args_train.tokenizer_max_length is not None:
        length_check.append(args_train.tokenizer_max_length)
    max_length = min(length_check)

    if compute_class_weights:
        num_classes_protected = get_class_weights(args_train.train_pkl, labels_prot_path, key_prot_list)
        protected_class_weights_list = [torch.tensor(n, device=device) for n in num_classes_protected]
    else:
        protected_class_weights_list = [None] * len_key_prot

    val_loader = get_data_loader(
        task_key=args_train.task_key,
        protected_key=key_prot_list,
        text_key=args_train.text_key,
        tokenizer=tokenizer,
        data_path=args_train.val_pkl,
        labels_task_path=args_train.labels_task_path,
        labels_prot_path=labels_prot_path,
        batch_size=args_train.batch_size,
        max_length=max_length,
        shuffle=False,
        debug=debug,
        only_load_dataset = return_dataset,
        external_data = external_data
    )

    test_loader = get_data_loader(
        task_key=args_train.task_key,
        protected_key=key_prot_list,
        text_key=args_train.text_key,
        tokenizer=tokenizer,
        data_path=args_train.test_pkl,
        labels_task_path=args_train.labels_task_path,
        labels_prot_path=labels_prot_path,
        batch_size=args_train.batch_size,
        max_length=max_length,
        shuffle=False,
        debug=debug,
        only_load_dataset = return_dataset,
        external_data = external_data
    )

    return test_loader, val_loader
