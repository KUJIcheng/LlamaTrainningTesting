import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk
import wandb
import os
import random
import numpy as np
#from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import deepspeed
from torch.cuda.amp import autocast


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["MASTER_ADDR"] = "localhost" 
os.environ["MASTER_PORT"] = "9994"
os.environ["NCCL_P2P_DISABLE"] = '1'

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

if torch.cuda.is_available():
    print("CUDA is available. Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize training hyperparameters
## I/O Paths
data_path = "/workspace/ML_team/datasets_4096/tokenized_data" # 改成对应的地址
model_path = './configs/config.json' # 如果使用1B，需要改成对应的json
checkpoint_output_dir = './model_checkpoints'
deepspeed_config = './config/test_ds_zero3_plus_config.json'
logging_dir = './logs'

## Training args
max_seq_len = 1024
attn_implementation = 'flash_attention_2'
eval_strategy = "steps"
vis_app = 'wandb'
save_strategy = 'no' # 不save model weight
logging_steps = 100
eval_steps = 100
num_epoch = 3
batch_size = 2
gradient_checkpointing = True
fp16 = True
learning_rate = 3e-4
gradient_accumulation = 16
weight_decay = 0.1 * learning_rate

# Wandb variables
wandb_key = 'ae05f44c8d5afe19940ef81e6f5cf69063392241'
project_name = 'llama-training'
entity_name = 'fjiang7-ucsd'


def initialize_model(config_path='./configs/config.json'):
    # the default config path is for llama 3.1 8b model
    with deepspeed.zero.Init():
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    return model

def main():

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("/workspace/ML_team/train/llama_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    model = initialize_model()
    model.config.use_cache=False
    model.to(device)
    #estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

    # load dataset, default to SlimPajama-62B
    #dataset = load_dataset('DKYoon/SlimPajama-6B', cache_dir='../hf_cache', split = ['train', 'validation'], download_mode="reuse_cache_if_exists")
    # approach 1 (save pre-processed data on disk and load) - require larger disk memory for large dataset
    dataset_train = load_from_disk(os.path.join("/workspace/ML_team/datasets_70_1024/tokenized_data", "train"))
    dataset_eval = load_from_disk(os.path.join("/workspace/ML_team/datasets_70_1024/tokenized_data", "validation"))
    
    # approach 2 (load raw from cache then tokenize, when data is large -- full 6B) - doesn't work, takes forever, ignore this
    #def tokenize_function(examples):
    #    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_len)
    
    #train_subset = load_dataset('DKYoon/SlimPajama-6B', cache_dir='/workspace/ML_team/hf_cache', split = 'train', download_mode='reuse_cache_if_exists')
    #eval_subset = load_dataset('DKYoon/SlimPajama-6B', cache_dir='/workspace/ML_team/hf_cache', split = 'validation', download_mode='reuse_cache_if_exists')
    #dataset_train = train_subset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=50)
    #dataset_eval = eval_subset.map(tokenize_function, batched=True, remove_columns=["text"], num_proc=50)

    #train_data = TextDataset(dataset_train, tokenizer, 2048)
    #eval_data = TextDataset(dataset_eval, tokenizer, 2048)

    wandb.login(key = wandb_key)  # Log in directly without setting env variable
    wandb.init(project=project_name, entity=entity_name)

    torch.cuda.empty_cache()  # Clear any residual GPU memory usage


    training_args = TrainingArguments(
        output_dir = checkpoint_output_dir,
        evaluation_strategy = eval_strategy,
        eval_steps = 100,
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size, 
        per_device_eval_batch_size = batch_size,
        num_train_epochs = num_epoch,
        weight_decay = weight_decay,
        gradient_accumulation_steps = gradient_accumulation,
        report_to = "wandb",
        logging_dir = logging_dir,
        logging_steps = logging_steps,
        lr_scheduler_type="cosine",
        save_steps = 500,
        deepspeed = deepspeed_config,
        fp16 = True,
        warmup_steps=500,
        gradient_checkpointing = True,
        save_strategy = save_strategy,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=data_collator
    )

    print("Start training...")
    trainer.train()

    model.save_pretrained("./final_model/target_model_config")
    tokenizer.save_pretrained("./final_model/target_tokenizer_config")
    print("Saved final model and tokenizer.")
    wandb.finish()


if __name__ == "__main__":
    main()