import torch
import torch.nn as nn
from mixout import MixLinear


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.linear1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(100, 10)
    
    def forward(self, input):
        input = self.drop1(self.relu1(self.linear1(input)))
        input = self.drop2(self.relu2(self.linear2(input)))
        return self.linear3(input) 
    
def main():
    # Prepare the model configuration from pretraining. In this example, 
    # I just use all one parameters as the pretraiend model configuration.
    model_config = {
                    'linear1.weight': torch.ones(300, 784), 'linear1.bias': torch.zeros(300),
                    'linear2.weight': torch.ones(100, 300), 'linear2.bias': torch.zeros(100),
                    'linear3.weight': torch.ones(10, 100), 'linear3.bias': torch.zeros(10)
                    }
    # Set up a model for finetuning.
    model = FullyConnected()
    print("Before applying mixout:")
    print(model)
    model.load_state_dict(model_config)
    
    # From now on, we are going to replace dropout with mixout.
    # Since dropout drops all parameters outgoing from the dropped neuron,
    # mixout mixes the parameters of the nn.Linear right after the nn.Dropout.
    for name, module in model.named_modules():
        if name in ['drop1', 'drop2'] and isinstance(module, nn.Dropout):
            setattr(model, name, nn.Dropout(0))
        if name in ['linear2', 'linear3'] and isinstance(module, nn.Linear):
            target_state_dict = module.state_dict()
            bias = True if module.bias is not None else False
            new_module = MixLinear(module.in_features, module.out_features, 
                                   bias, target_state_dict['weight'], 0.9)
            new_module.load_state_dict(target_state_dict)
            setattr(model, name, new_module)
    print("After applying mixout")
    print(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model.to(device=device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    
    for _ in range(10):
        optimizer.zero_grad()
        x = torch.randn(16, 784).to(device)
        y = torch.ones(16, dtype=torch.long).to(device)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        print(loss.item())
    
if __name__ == "__main__":
    main()
    
