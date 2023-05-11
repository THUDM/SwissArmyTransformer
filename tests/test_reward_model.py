import torch

from sat import get_args, get_tokenizer, AutoModel, training_main
from sat.rlhf.reward_model import RewardModel

if __name__ == "__main__":
    # Parse args, initialize the environment. This is necessary.
    args = get_args()
    # Automatically download and load model. Will also dump model-related hyperparameters to args.
    model, args = AutoModel.from_pretrained('bert-base-uncased', args) 
    # Get the BertTokenizer according to args.tokenizer_type (automatically set).
    tokenizer = get_tokenizer(args) 

    reward_model = RewardModel(
        base_model = model,
        args=args
    )
    
    text = [["This is a piece of text.", "Another piece of text."], ["This is a piece of text.", "Another piece of text."]]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])

    encoded_input.to("cuda")
    reward_model.to("cuda")

    print(encoded_input.keys())
    print(encoded_input["input_ids"].shape)
    print(encoded_input["token_type_ids"].shape)
    print(encoded_input["attention_mask"].shape)

    swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), token_type_ids=encoded_input['token_type_ids'].cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
    print(swiss_output)

    reward_output = reward_model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda(), token_type_ids=encoded_input['token_type_ids'].cuda())
    print(reward_output)

    value_output = reward_model.forward_value(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda(), token_type_ids=encoded_input['token_type_ids'].cuda(), prompt_length=9)
    print(value_output)
