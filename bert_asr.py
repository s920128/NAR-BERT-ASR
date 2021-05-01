import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, MultiheadAttention, Linear, Dropout, LayerNorm, ModuleList
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertPreTrainedModel, AdamW
from transformers.modeling_bert import *
from typing import Optional, Any


class BasicPositionalEncoding(nn.Module):
    def __init__(self, d_model=768, dropout=0.1, max_len=5000):
        super(BasicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """
    def __init__(self, idim, odim = 768, dropout_rate=0.1, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(32 * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else BasicPositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask=None):
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=768, dropout=0.1, max_len=60):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self):
        #x: [seq, batch ,d]
        x = self.pe[:, :]
        return self.dropout(x)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class PDSLayer(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(PDSLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward//2, d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = F.glu
        
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
    
        
class PDS(nn.Module):
    def __init__(self, decoder_layer, num_layers=4, norm=None):
        super(PDS, self).__init__()
        self.position_multihead_attn = MultiheadAttention(768, 8, dropout=0.1)
        self.norm1 = LayerNorm(768)
        self.linear1 = Linear(768, 2048)
        self.dropout = Dropout(0.1)
        self.linear2 = Linear(2048//2, 768)
        self.activation = F.glu
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)
        self.layers = _get_clones(decoder_layer, num_layers-1)
        self.num_layers = num_layers-1
        self.norm = norm
    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        tgt2 = self.position_multihead_attn(tgt, memory, memory, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        #######
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class TransformerEncoderLayerPre(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayerPre, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward//2, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = F.glu
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

class BertForMaskedLMForBERTASR(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = LabelSmoothingCrossEntropy()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class BERTASR_Encoder(nn.Module):
    def __init__(self, idim, odim, attention_dim=768, attention_heads=8, dropout_rate=0.1):
        super(BERTASR_Encoder, self).__init__()
        self.model_type = 'Transformer'
        self.odim = odim
        self.embed = Conv2dSubsampling(idim, attention_dim, dropout_rate)
        self.pe = PositionalEncoding(attention_dim)
        encoder_layers = TransformerEncoderLayerPre(d_model=attention_dim, nhead=attention_heads, dropout=dropout_rate)
        self.encoder_norm = nn.LayerNorm(attention_dim)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=6, norm= None)
        pdslayer = PDSLayer(d_model=attention_dim, nhead=attention_heads, dim_feedforward=2048, dropout=dropout_rate)
        self.pds = PDS(pdslayer, num_layers=4)
        self.decoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.classifier = nn.Linear(768, odim, bias=False)

    def forward(self, src, label=None):
        #src: [batch, time, idim]
        #conv2D
        src = self.embed(src)
        #convert to [time, batch, idim] (For Pytorch transformer)
        src = src.transpose(0,1)
        src = self.encoder_norm(self.encoder(src))
        q = self.pe().repeat(1,src.shape[1],1)
        src = self.pds(q, src)
        src = self.decoder(src)
        #convert to [batch, time, idim]
        src = src.transpose(0,1)
        #classification
        output = self.classifier(src)
        if label !=None:
            loss_fct = LabelSmoothingCrossEntropy()
            loss = loss_fct(output.view(-1, self.odim), label.view(-1))
            return loss
        else:
            return output

class BERTASR(nn.Module):
    def __init__(self, encoder, BertMLM):
        super(BERTASR, self).__init__()
        self.encoder = encoder
        self.bertmodel = BertMLM

    def forward(self, src, label=None):
        #Conv
        src = self.encoder.embed(src)
        #convert to [time, batch, idim] (For Pytorch transformer)
        src = src.transpose(0, 1)
        src = self.encoder.encoder_norm(self.encoder.encoder(src))
        q = self.encoder.pe().repeat(1, src.shape[1], 1)
        src = self.encoder.pds(q, src)
        src = self.encoder.decoder(src)
        #convert to [batch, time, idim]
        src = src.transpose(0, 1)
        # TO BERT Classification
        return self.bertmodel(inputs_embeds=src, labels=label)

def avg_model(root="./pretraining.", avg_num=10, last_num=130, save_path="./pretraining_avg10"):
    avg=None
    # sum
    for num in range(0,avg_num):
        states = torch.load(root+str(last-num), map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= avg_num
            else:
                avg[k] //= avg_num
    torch.save(avg, save_path)