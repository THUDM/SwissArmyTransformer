import os
from .utils import *

class BaseTokenizer:
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, text, **kwargs):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, **kwargs)
    
    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def __repr__(self):
        """info interpretation for tokenizer"""
        return "Base Tokenizer for SAT"
    
    @property
    def command_tokens(self):
        """get command tokens of the tokenizer"""
        return None
    
    @property
    def num_tokens(self):
        """get total number of tokens"""
        return 0
    
    def from_pretrained(self, **kwargs):
        """load tokenizer params from pretrained"""
        pass
    
    def EncodeAsIds(self, text, **kwargs):
        """encode to ids by tokenizer"""
        raise NotImplementedError
    
    def EncodeAsTokens(self, text, **kwargs):
        """encode to tokens by tokenizer"""
        raise NotImplementedError
        
    def DecodeIds(self, ids, **kwargs):
        """decode ids to original form by tokenizer"""
        raise NotImplementedError
    
    def DecodeTokens(self, tokens, **kwargs):
        """decode tokens to original form by tokenizer"""
        raise NotImplementedError

    
        
        
    