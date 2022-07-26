from .utils import cond_log_prob, generate_text


class ModelForEvaluation:
    def __init__(self, model):
        self.model = model

    def cond_log_prob(self, batch) -> list[float]:
        """
        @return: Conditional log probability of each option
        """
        self.model.eval()
        return cond_log_prob(self.model, batch)

    def generate_text(self, batch, strategy, max_length) -> list[list[int]]:
        """
        @return: A list of text model generated, sorted by score in descending order
        """
        self.model.eval()
        return generate_text(self.model, batch, strategy, max_length)
