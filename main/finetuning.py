import json
import time
import torch
import pathlib
import tiktoken
from data_preprocessing import create_data_loader, format_input
from gpt.build_llm.gpt2 import GPTModel
from gpt.build_llm.utils import train_model
from gpt.Inference.generate_logic import generate_and_print_sample
from gpt.model_weights.load_model_weights import load_weights,load_foundational_model

model_type = "gpt2-medium"
tokenizer = tiktoken.get_encoding("gpt2")
current_path = pathlib.Path(__file__).resolve().parent.parent
with open('../base_config.json', 'r') as f:
    configs = json.load(f)
    GPT_CONFIG = configs["base_configs"]
    GPT_CONFIG.update(configs["model_configs"][model_type])

with open(current_path/'datasets/instruction-data.json', 'r') as f:
    data = json.load(f)

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Converting raw data to a dataloader object
"""
train_loader = create_data_loader(train_data, batch_size=4,device = device,shuffle=True, drop_last=True,num_workers=0)
val_loader = create_data_loader(val_data, batch_size=4,device = device,shuffle=False, drop_last=False,num_workers=0)
test_loader = create_data_loader(test_data, batch_size=4,device = device,shuffle=False, drop_last=False,num_workers=0)

model = GPTModel(GPT_CONFIG)
"""
loading the pretrained model weights
"""
state_dict = load_foundational_model(model_type)
model = load_weights(GPT_CONFIG["n_layers"], model, state_dict)

# """
# Fine tuning then model on instruction dataset
# """
#
# start_time = time.time()
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
# num_epochs = 5
# model.to(device)
# train_losses, val_losses, tokens_seen = train_model(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#     start_context=format_input(val_data[0]), tokenizer=tokenizer
# )
#
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")

"""
Testing the model by predicting on test data
"""
entry = test_data[0]
input_text = format_input(entry)
generate_and_print_sample(model, tokenizer, device, input_text,max_new_tokens=256,temperature=0.0,top_k=None,eos_id=50256)





