import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk
import wandb
import os
import random
import numpy as np

# uncommented this, installed deepspeed since it wasn't installed in ML_team (!!)
import deepspeed
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

# packages added for flash attention (!!)
from monkey_patch_packing import monkey_patch_packing_for_model
from packed_dataset import PackedDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # changed to train on one GPU only (!!)

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
data_path = "/workspace/ML_team/datasets_4096/tokenized_data"
model_path = '../configs/config_1b/config.json' # changed this for funsies idk (!!)
checkpoint_output_dir = './model_checkpoints'
deepspeed_config = './config/test_ds_zero3_plus_config.json'
logging_dir = './logs'

## implementing monkey patch (!!)
monkey_patch_packing_for_model(model_path)

## Training args
attn_implementation = 'flash_attention_2'
eval_strategy = "steps"
vis_app = 'wandb'
save_strategy = 'no'
logging_steps = 100
eval_steps = 100
num_epoch = 3
batch_size = 4 # changed to 4 to bypass error maybe (!!)
gradient_checkpointing = True
fp16 = True
learning_rate = 3e-4
gradient_accumulation = 16
weight_decay = 0.1 * learning_rate

# Wandb variables
wandb_key = 'ae05f44c8d5afe19940ef81e6f5cf69063392241'
project_name = 'fa_test' # named fa_test to compare results in wandb (!!)
entity_name = 'fjiang7-ucsd'


def initialize_model(config_path='../configs/config.json'):
    # changed config path to use 1b for testing (!!)
    with deepspeed.zero.Init():
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        model.config.use_cache = False # added this per instructions (!!)
    return model


def main():

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("/workspace/ML_team/train/llama_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    # ran into deepspeed issues so I decided not to use it for now to focus on FA implementation
    # model = initialize_model()
    config = AutoConfig.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_config(
        config, 
        torch_dtype=torch.bfloat16,
        attn_implementation = 'flash_attention_2')
    
    model.config.use_cache=False
    model.to(device)
    #estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

    # load dataset, default to SlimPajama-62B
    #dataset = load_dataset('DKYoon/SlimPajama-6B', cache_dir='../hf_cache', split = ['train', 'validation'], download_mode="reuse_cache_if_exists")
    dataset_train = load_from_disk(os.path.join("/workspace/ML_team/datasets/tokenized_data", "train"))
    dataset_eval = load_from_disk(os.path.join("/workspace/ML_team/datasets/tokenized_data", "validation"))
    
    # something something debugging (!!)

    # Before packing the dataset
    print(f"Number of training examples: {len(dataset_train)}")
    print(f"Number of validation examples: {len(dataset_eval)}")

    # Check a few data points
    for i, data_point in enumerate(dataset_train):
        print(f"Data point {i}: keys: {list(data_point.keys())}")
        print(f"Input IDs length: {len(data_point['input_ids'])}")
        print(f"Attention Mask shape: {getattr(data_point['attention_mask'], 'shape', 'N/A')}")
        if i >= 2:  # Limit to a few prints
            break


    # packing the datasets (!!)
    pack_length = 8192 # can be changed, must be >= max_position_embeddings; currently equal (!!) 
    packed_train_dataset = PackedDataset(dataset_train, tokenizer, pack_length)
    packed_eval_dataset = PackedDataset(dataset_eval, tokenizer, pack_length)

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
        #save_steps = 500,
        
        # removed to not use deepspeed (!!)
        #deepspeed = deepspeed_config,

        fp16 = True,
        warmup_steps=200,
        gradient_checkpointing = True,
        save_strategy = save_strategy,
        save_total_limit=2,
    )

    # getting rid of this since we are using sequence packing (!!)
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=packed_train_dataset, # changed to packed dataset (!!)
        eval_dataset=packed_eval_dataset, # changed to packed dataset (!!)

        # getting rid of this since we are using sequence packing (!!)
        # data_collator=data_collator
    )

    # Debugging step before the model is trained
    for batch in trainer.get_train_dataloader():
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        assert batch['input_ids'].dim() >= 1, "input_ids is not at least 1D!"
        assert batch['attention_mask'].dim() >= 1, "attention_mask is not at least 1D!"
        break

    print("Start training...")
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()