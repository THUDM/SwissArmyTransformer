from transformers import T5Tokenizer
from .glm.tokenization import Tokenization, CommandToken


class HFTokenizer:
    def __init__(self, model_cls, model_type_or_path=None, cache_dir=None, command_tokens=None):
        self.text_tokenizer = model_cls.from_pretrained(model_type_or_path, cache_dir=cache_dir)
        self.num_tokens = len(self.text_tokenizer)
        self._command_tokens = []
        self.command_name_map = {}
        self.command_token_map = {}
        self.command_id_map = {}

    @property
    def command_tokens(self):
        return self._command_tokens

    @command_tokens.setter
    def command_tokens(self, command_tokens):
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self.command_tokens}
        self.command_token_map = {tok.token: tok for tok in self.command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self.command_tokens}

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    def EncodeAsIds(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        ids = self.text_tokenizer.encode(processed_text, add_special_tokens=False)
        tokenization = Tokenization(ids, processed_text, text)
        return tokenization

    def DecodeIds(self, ids):
        if isinstance(ids, Tokenization):
            ids = ids.tokenization
        return self.text_tokenizer.decode(ids)

    def DecodeTokens(self, tokens):
        return self.text_tokenizer.convert_tokens_to_string(tokens)

    def IdToToken(self, Id):
        if isinstance(Id, CommandToken):
            return Id.token
        return self.text_tokenizer.convert_ids_to_tokens(Id)

    def TokenToId(self, token):
        if isinstance(token, CommandToken):
            return token.Id
        return self.text_tokenizer.convert_tokens_to_ids(token)


class HFT5Tokenizer(HFTokenizer):
    def __init__(self, model_type_or_path=None, cache_dir=None):
        super().__init__(T5Tokenizer, model_type_or_path=model_type_or_path, cache_dir=cache_dir)
        command_tokens = [
            CommandToken('eos', '</s>', self.TokenToId("</s>")),
            CommandToken('pad', '<pad>', self.TokenToId("<pad>")),
        ]
        for i in range(100):
            command_tokens.append(CommandToken(f'MASK{i}', f'<extra_id_{i}>', self.TokenToId(f'<extra_id_{i}>')))
        self.command_tokens = command_tokens
