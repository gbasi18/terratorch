
import warnings
import re
import torch
import numpy as np
from einops import rearrange
from torch import nn
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import logging





def build_blank_wordpiece():
    # Minimal tokens needed for testing
    vocab = {"[PAD]": 0, "[UNK]": 1, "[SOS]": 2,"[EOS]": 3, "[S_0]": 4, "[S_1]": 5, "[S_2]": 6}
    added_tokens = [AddedToken(w, normalized=False, special=True) for w in vocab.keys()]
    tok = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tok.normalizer = BertNormalizer()
    tok.pre_tokenizer = BertPreTokenizer()
    tok.decoder = WordPieceDecoder()
    tok.add_tokens(added_tokens)
    return tok


def capitalize_sentences(text):
    # Split text into sentences using a regex that looks for sentence end punctuation
    sentences = re.split('([.!?] *)', text)
    capitalized = ''.join([s.capitalize() for s in sentences])
    return capitalized


class CaptionTokenizer(nn.Module):
    def __init__(self, tokenizer_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer_file is not None:
            self.text_tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.text_tokenizer = build_blank_wordpiece()  # un-trained
        self.text_tokenizer.enable_padding()

    def encode(self, text: list[str], device: torch.device, eos_id=3, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Args:
            text list[str]: Text to be tokenized
            device: torch.device
        Returns:
            dict for generation sampler input
        """
        # Add start token
        text = [t + " [S_1]" for t in text]

        # Tokenize
        tok_ids = [t.ids for t in self.text_tokenizer.encode_batch(text, add_special_tokens=True)]

        # Add EOS token
        tok_ids = [t + [eos_id] for t in tok_ids]


        tok_ids = torch.tensor(tok_ids, device=device)

        text_dict = {
            "tensor": tok_ids,
            "input_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
            "target_mask": torch.ones_like(tok_ids, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
        }

        return text_dict

    def decode_text(self, mod_dict, key="caption"):
        """
        Decodes a text sequence from a model dictionary.

        Args:
            mod_dict (dict): Model output dictionary.
            key (str): Key of the text modality to decode.
        """
        decoded_texts = []

        for i in range(mod_dict[key]["tensor"].shape[0]):
            seq = mod_dict[key]["tensor"][i]
            seq = seq[mod_dict[key]["input_mask"][i] == 0]
            seq = seq.tolist()

            merged_text = self.text_tokenizer.decode(seq, skip_special_tokens=False)

            decoded_texts.append(merged_text.replace(" [EOS]", ""))

        decoded_texts = list(map(capitalize_sentences, decoded_texts))

        return decoded_texts


class CoordsTokenizer(nn.Module):
    def __init__(self, tokenizer_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer_file is not None:
            self.text_tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            self.text_tokenizer = build_blank_wordpiece()  # un-trained

    def encode(self, coords: torch.Tensor, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Encodes coords to token ids. Returns tuple to be compatible with image tokenizers.

        Args:
            coords (torch.Tensor): Center coordinates of image with shape [B, 2] with [lon, lat] values in second dim.

        Returns:
            tok_ids(tuple[torch.Tensor]): Token ids with shape [B, 2]
        """
        if coords.shape[1] != 2:
            raise ValueError(f"Expect coords data in shape [batch, 2] with [lon, lat] values, "
                             f"got coords with shape {coords.shape}.")

        # Align coords with 0.25 degree grid
        coords = (coords * 4).round() / 4
        device = coords.device

        coords = [f"lat={c[1].item():.2f} lon={c[0].item():.2f} [EOS]" for c in coords]

        # Tokenize
        tok_ids = [t.ids for t in self.text_tokenizer.encode_batch(coords, add_special_tokens=True)]
        tok_ids = torch.tensor(tok_ids, device=device)

        coords_dict = {
            "tensor": tok_ids,
            "input_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
            "target_mask": torch.ones_like(tok_ids, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
        }

        return coords_dict

    def decode_text(self, mod_dict, key="coords"):
        """
        Decodes a coordinate sequence from a modality dictionary.

        Args:
            mod_dict (dict): Model output dictionary.
            key (str): Key of the coords modality to decode.
        """
        coords = []

        B = mod_dict[key]["tensor"].shape[0]

        for i in range(B):
            seq = mod_dict[key]["tensor"][i].tolist()[:2]

            text = self.text_tokenizer.decode(seq, skip_special_tokens=False)

            try:
                lat, lon = text.split(" ")
                coords.append([float(lon.strip("lon=")), float(lat.strip("lat="))])
            except Exception as e:
                warnings.warn(f"Coordinate generation did not work correctly, generated text: {text} (Error: {e}). "
                              f"Returning NaN.")
                coords.append([torch.nan, torch.nan])

        return coords

#! PRECIP_VARIABLES_DAILY = ["prec_min", "prec_max", "prec_sum"]
#! TEMP_VARIABLES_DAILY   = ["t_min", "t_max", "t_mean"] those get concatenated e.g ["t_min = lower --> upper",...,"t_max = lower --> upper",....,"t_mean = lower --> upper"]

#! PRECIP_VARIABLES = ["prec"]
#! TEMP_VARIABLES   = ["temp"] # this is presented 



class WeatherTokenizer(nn.Module):
    """
    Vectorise / de-vectorise weather time-series.
    """

    _TOKEN_MAP = {}
    _UNK_ID = 1

    @classmethod
    def _build_token_map(cls, tok_path: str) -> None:
        """Parse the tokenizer file once and cache all variables.
        
        Each token of the form  "var=lower-->higher"  defines one closed open bin.
        Example:  temp=10-->20   applies for  10 < x <= 20 .
        """
        if tok_path in cls._TOKEN_MAP:
            return                                

        tk = Tokenizer.from_file(tok_path)
        vocab = tk.get_vocab()

        buckets = {}
        for token, tid in vocab.items():
            if "-->" not in token:
                continue
            var, rng = token.split("=", 1)
            _, hi = map(float, rng.split("-->"))
            buckets.setdefault(var, []).append((hi, tid))

    
        token_map: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for var, pairs in buckets.items():
            pairs.sort(key=lambda p: p[0])
            upp, ids = zip(*pairs)
            token_map[var] = (torch.tensor(upp), torch.tensor(ids))

        cls._TOKEN_MAP[tok_path] = token_map
        
    def __init__(self, tokenizer_path: str | None, variables: List[str]):
        super().__init__()
        self.variables = variables
        self.tok_path = tokenizer_path  

        if tokenizer_path:
            self.text_tokenizer = Tokenizer.from_file(tokenizer_path)
            self._build_token_map(tokenizer_path)
        else:
            self.text_tokenizer = self._build_blank_wordpiece()
            

    def _values_to_ids(self, values: torch.Tensor, var: str) -> torch.Tensor:
        """
        Convert a 1-D tensor of float values to token ids for one variable.
        """
        if self.tok_path is None or var not in self._TOKEN_MAP[self.tok_path]:                       
            return torch.full_like(values, self._UNK_ID, dtype=torch.long)

        uppers, ids = self._TOKEN_MAP[self.tok_path][var]
        if uppers.numel() == 0:
            return torch.full_like(values, self._UNK_ID, dtype=torch.long)

        idx = torch.bucketize(values, uppers, right=True)
        unk = idx == len(uppers)
        idx.clamp_(max=len(uppers) - 1)
        out = ids[idx]
        out[unk] = self._UNK_ID
        return out

    def encode(self, weather: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encode weather time-series -> token ids.
        Args:
            weather (torch.Tensor): [batch, seq_len, n_variables] Possible variables ["temp"], ["t_min","t_max","t_mean] or any subset of the last list
        """
       
        if weather.ndim != 3:
            raise ValueError("weather must be [batch, seq_len, n_vars]")
        device = weather.device
        b, s, _ = weather.shape
        chunks: List[torch.Tensor] = []
        for i, var in enumerate(self.variables):
            flat = rearrange(weather[:, :, i], "b s -> (b s)")
            ids = self._values_to_ids(flat, var)
            chunks.append(rearrange(ids, "(b s) -> b s", b=b, s=s))

        tok_ids = torch.cat(chunks, dim=1)
        weather_dict = {
            "tensor": tok_ids,  #!Token IDs with shape [batch, seq_len * n_variables] orther given by the list of
            "input_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
            "target_mask": torch.ones_like(tok_ids, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros_like(tok_ids, dtype=torch.bool, device=device),
        }
        return weather_dict

    @staticmethod
    def _parse_weather_token(token: str) -> float:
        if token == "[ZERO]":
            return 0.0
        if token == "[PAD]":
            return np.nan
        try:
            lo, hi = map(float, token.split("=", 1)[1].split("-->"))
            return (lo + hi) / 2.0
        except Exception:
            return np.nan

    @classmethod
    def _tokens_to_numeric(cls, tokens: List[str]) -> np.ndarray:
        return np.array([cls._parse_weather_token(t) for t in tokens])

    def decode_text(self, mod_dict: dict, key: str = "temperature_daily") -> List[List[float]]: #! key = "temperature_hourly"
        out: List[List[float]] = []
        for seq_ids in mod_dict[key]["tensor"].tolist():
            tokens = self.text_tokenizer.decode(seq_ids, skip_special_tokens=False).split()
            out.append(self._tokens_to_numeric(tokens).tolist())
        return out
