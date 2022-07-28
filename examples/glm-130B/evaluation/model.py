import torch

from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence

from .utils import process_data


class ModelForEvaluation(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def cond_log_prob(self, batch) -> list[float]:
        """
        @return: Conditional log probability of each option
        """
        tokens, targets, position_ids, attention_mask = process_data(batch)

        self.model.eval()
        with torch.no_grad():
            logits, *output_per_layers = self.model(tokens, position_ids, attention_mask, log_attention_weights=None)

        # output: [b, sq, vocab]
        output = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_ids = torch.arange(tokens.size(0), dtype=tokens.dtype, device=tokens.device).unsqueeze(1)

        choice_logits = []

        # Single token
        if batch["is_single_token"].any():
            target_ids = batch["choice_target_ids"][0]
            logits = output[batch_ids, target_ids, batch["choices"]]
            choice_logits = logits.squeeze(0).tolist()
        # Multi token
        else:
            for target_ids in batch["choice_target_ids"]:
                logits = output[batch_ids, target_ids, targets[batch_ids, target_ids]]
                choice_logits.append(logits.squeeze(0).sum(dim=-1).tolist())

        return choice_logits

    def generate_text(self, batch, strategy, max_length) -> list[list[int]]:
        """
        @return: A list of text model generated, sorted by score in descending order
        """

        seq = torch.squeeze(batch["tokens"].to(device=torch.cuda.current_device()).long())[:max_length]
        context_length = batch["context_length"].to(device=torch.cuda.current_device()).long()
        seq[context_length:] = -1

        def get_masks_and_position_ids(seq):
            tokens = seq.unsqueeze(0)
            attention_mask = batch["attention_mask"].to(device=torch.cuda.current_device()).bool().unsqueeze(1)
            position_ids = batch["position_ids"].to(device=torch.cuda.current_device()).long()
            return tokens, attention_mask, position_ids

        self.model.eval()
        with torch.no_grad():
            output = filling_sequence(
                self.model,
                seq,
                get_masks_and_position_ids=get_masks_and_position_ids,
                batch_size=strategy.num_beams if hasattr(strategy, "num_beams") else 1,
                strategy=strategy,
            )[0]

        if isinstance(output, torch.Tensor):  # different strategies
            output = list(output)

        output_targets = []

        for line in output:
            line = line.tolist()
            unfinished = line.index(-1) if -1 in line else len(line)
            if line[unfinished - 1] in strategy.end_tokens:
                unfinished -= 1
            line = line[context_length:unfinished]
            output_targets.append(line)

        return output_targets
