from transformers import (
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    RoFormerTokenizer,
)


def create_dataset(datasets, palm_type="VHHBERT"):
    assert palm_type in (
        "VHHBERT",
        "VHHBERT-w/o-PT",
        "AbLang",
        "AntiBERTa2",
        "AntiBERTa2-CSSP",
        "ESM-2-150M",
        "ESM-2-650M",
        "IgBert",
        "ProtBert",
    )
    if palm_type in ["VHHBERT", "VHHBERT-w/o-PT"]:
        tokenizer = BertTokenizerFast.from_pretrained("COGNANO/VHHBERT")
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence", "token_type_ids"])
        return tokenized_datasets, tokenizer
    elif palm_type == "AbLang":
        tokenizer = BertTokenizerFast(
            vocab_file="./benchmarks/data/vocab_ablang.txt",
            do_lower_case=False,
            do_basic_tokenize=False,
            unk_token="<unk>",
            sep_token=">",
            pad_token="-",
            cls_token="<",
            mask_token="*",
            tokenize_chinese_chars=False,
        )
        # Trim the sequence to fit the max length 160 of the positional embedding layer
        datasets = datasets.map(lambda examples: {"VHH_sequence": examples["VHH_sequence"][10:]})
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence", "token_type_ids"])
        return tokenized_datasets, tokenizer
    elif palm_type == "AntiBERTa2":
        tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence", "token_type_ids"])
        return tokenized_datasets, tokenizer
    elif palm_type == "AntiBERTa2-CSSP":
        tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2-cssp")
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence", "token_type_ids"])
        return tokenized_datasets, tokenizer
    elif palm_type == "ESM-2-150M":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence"])
        return tokenized_datasets, tokenizer
    elif palm_type == "ESM-2-650M":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence"])
        return tokenized_datasets, tokenizer
    elif palm_type == "IgBert":
        tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence", "token_type_ids"])
        return tokenized_datasets, tokenizer
    elif palm_type == "ProtBert":
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        tokenized_datasets = _preprocess(datasets, tokenizer, ["VHH_sequence", "token_type_ids"])
        return tokenized_datasets, tokenizer


def _preprocess(datasets, tokenizer, remove_columns):
    tokenized_datasets = datasets.map(
        lambda examples: {"VHH_sequence": " ".join(examples["VHH_sequence"])}
    )
    tokenized_datasets = tokenized_datasets.map(
        lambda examples: tokenizer(examples["VHH_sequence"]), batched=True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(remove_columns)
    return tokenized_datasets
