import torch.nn as nn

class DistillModel(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    def forward(self, teacher_kwargs, student_kwargs):
        teacher_logits, *mem_t = self.teacher(**teacher_kwargs)
        student_logits, *mem_s = self.student(**student_kwargs)
        return teacher_logits, student_logits
    
    def disable_untrainable_params(self):
        for n, p in self.teacher.named_parameters():
            p.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT-distill', 'BERT distill Configurations')
        group.add_argument('--teacher', type=str)
        return parser