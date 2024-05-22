import ablang
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    BertModel,
    RobertaConfig,
    RobertaModel,
    RoFormerModel,
)

MAX_LENGTH = 185


class PalmForBindingPrediction(nn.Module):

    def __init__(self, palm_type="VHHBERT"):
        super().__init__()
        assert palm_type in (
            "VHHBERT",
            "VHHBERT-w/o-PT",
            "AbLang",
            "AntiBERTa2",
            "AntiBERTa2-CSSP",
            "ESM-2",
            "IgBert",
            "ProtBert",
        )
        if palm_type == "VHHBERT":
            self.palm = VhhBert()
        elif palm_type == "VHHBERT-w/o-PT":
            self.palm = VhhBertWithoutPretrain()
        elif palm_type == "AbLang":
            self.palm = AbLang()
        elif palm_type == "AntiBERTa2":
            self.palm = AntiBERTa2()
        elif palm_type == "AntiBERTa2-CSSP":
            self.palm = AntiBERTa2CSSP()
        elif palm_type == "ESM-2":
            self.palm = ESM2()
        elif palm_type == "IgBert":
            self.palm = IgBert()
        elif palm_type == "ProtBert":
            self.palm = ProtBert()
        self.classifier = ClassificationHead(palm_type)

    def forward(self, input_ids=None, attention_mask=None, antigen_embeddings=None, labels=None):
        vhh_embeddings = self.palm(input_ids, attention_mask)
        embeddings = torch.cat([vhh_embeddings, antigen_embeddings], dim=1)
        logits = self.classifier(embeddings)
        logits = logits.squeeze(-1)
        output = {"logits": logits}
        bce = nn.BCEWithLogitsLoss()
        loss = bce(logits, labels.float())
        output["loss"] = loss
        return output


class VhhBert(nn.Module):

    def __init__(self):
        super().__init__()
        self.vhhbert = RobertaModel.from_pretrained("tsurubee/VHHBERT")

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.vhhbert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class VhhBertWithoutPretrain(nn.Module):

    def __init__(self):
        super().__init__()
        config = RobertaConfig(
            vocab_size=25,
            max_position_embeddings=MAX_LENGTH,
        )
        self.vhhbert = RobertaModel(config)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.vhhbert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class AbLang(nn.Module):

    def __init__(self):
        super().__init__()
        ablang_h = ablang.pretrained("heavy")
        ablang_h.unfreeze()
        self.AbRep = ablang_h.AbRep

    def forward(self, input_ids=None, attention_mask=None):
        last_hidden_states = self.AbRep(
            input_ids, attention_mask=attention_mask
        ).last_hidden_states
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class AntiBERTa2(nn.Module):

    def __init__(self):
        super().__init__()
        self.AntiBERTa2 = RoFormerModel.from_pretrained("alchemab/antiberta2")

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.AntiBERTa2(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class AntiBERTa2CSSP(nn.Module):

    def __init__(self):
        super().__init__()
        self.AntiBERTa2 = RoFormerModel.from_pretrained("alchemab/antiberta2-cssp")

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.AntiBERTa2(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class IgBert(nn.Module):

    def __init__(self):
        super().__init__()
        self.IgBert = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.IgBert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class ProtBert(nn.Module):

    def __init__(self):
        super().__init__()
        self.ProtBert = BertModel.from_pretrained("Rostlab/prot_bert")

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.ProtBert(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class ESM2(nn.Module):

    def __init__(self):
        super().__init__()
        self.ESM2 = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.ESM2(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        vhh_embeddings = _mean_embeddings(last_hidden_states, attention_mask)
        return vhh_embeddings


class ClassificationHead(nn.Module):

    def __init__(self, palm_type):
        super().__init__()
        assert palm_type in (
            "VHHBERT",
            "VHHBERT-w/o-PT",
            "AbLang",
            "AntiBERTa2",
            "AntiBERTa2-CSSP",
            "ESM-2",
            "IgBert",
            "ProtBert",
        )
        if palm_type in ["AntiBERTa2", "AntiBERTa2-CSSP", "IgBert", "ProtBert"]:
            embedding_dim = 1024
        elif palm_type == "ESM-2":
            embedding_dim = 640
        else:
            embedding_dim = 768
        self.dense = nn.Linear(embedding_dim + 640, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def _mean_embeddings(hidden_states, attention_mask):
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
    sum_mask = attention_mask_expanded.sum(1)
    return sum_embeddings / sum_mask
