import fire
import random
import torch
import torch.optim as optim
import torch.nn as nn
from peft import get_peft_model, prepare_model_for_int8_training
from torch.optim.lr_scheduler import StepLR
from transformers import LlamaTokenizer
from  LLama import (
    LlamaForCausalLM,
)

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from trainingconfig import train_config as TRAIN_CONFIG
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
)
from concatenator import get_dataloader_kwargs
from llama_recipes.utils.train_utils import (
    print_model_size,
    setup,
)

from MyDataset.IncontextData import InContextDataTrain

import time
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
from llama_recipes.utils.memory_utils import MemoryTrace


def load_model(kwargs, train_config):
    if kwargs['Model'] == "LLama-7B":
        train_config.model_name = 'meta-llama/Llama-2-7b-chat-hf'
    elif kwargs['Model'] == "LLama-13B":
        train_config.model_name = 'meta-llama/Llama-2-13b-chat-hf'
    elif kwargs['Model'] == "Vicuna-7B":
        train_config.model_name = 'lmsys/vicuna-7b-v1.5'
    elif kwargs['Model'] == "Vicuna-13B":
        train_config.model_name = 'lmsys/vicuna-13b-v1.5'

    if kwargs['Model'] == "LLama-7B":
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )
    elif kwargs['Model'] == "LLama-13B":
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )
    elif kwargs['Model'] == "Vicuna-7B":
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )
    elif kwargs['Model'] == "Vicuna-13B":
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )   

 
    if kwargs['Model'] == "LLama-7B":
        model_gold = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )
    elif kwargs['Model'] == "LLama-13B":
        model_gold = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )
    elif kwargs['Model'] == "Vicuna-7B":
        model_gold = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )
    elif kwargs['Model'] == "Vicuna-13B":
        model_gold = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache = False,
        )

    # Load the tokenizer and add special tokens
    if kwargs['Model'] == "LLama-7B":
        tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif kwargs['Model'] == "LLama-13B":
        tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif kwargs['Model'] == "Vicuna-7B":
        tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif kwargs['Model'] == "Vicuna-13B":
        tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, model_gold, tokenizer


def train(model, model_gold, train_dataloader, gold_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, train_config):
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler() 
 
    train_prep = []
    train_loss = []
    epoch_times = []
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(gold_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length)
            for (step, (batch, batch_gold)) in enumerate(zip(train_dataloader, gold_dataloader)):
                for key in batch_gold.keys():
                    batch_gold[key] = batch_gold[key].to("cuda")
                    batch[key] = batch[key].to("cuda")
                
                output = model(**batch)
                with torch.no_grad():
                    outputgold = model_gold(**batch_gold)
                Attn_States = output.Attn_states
                Attn_States_gold = outputgold.Attn_states

                Attn_States = torch.stack(Attn_States, dim=1)
                Attn_States_gold = torch.stack(Attn_States_gold, dim=1)

                batch_size = int(batch_gold.data['input_ids'].shape[0])
                HiddenSize = Attn_States.shape[-1]
                gold = list()
                
                loss_contrast = 0
                numPerSam = batch_size
                for i in range(0, batch_size, numPerSam):
                    senGold = dict()
                    senOri = dict()
                    senGold['0'] = dict()
                    senOri['0'] = dict()
                    for m in range(10):
                        senGold[str(m)] = dict()
                        senOri[str(m)] = dict()
                    
                    for j in range(numPerSam):
                        permutationTmp = batch.data['permutation'][i+j]
                        for m in range(permutationTmp.shape[0]-1):
                            if str(int(permutationTmp[m])) in senOri.keys():
                                if str(m) not in senGold[str(int(permutationTmp[m]))].keys():
                                    senOri[str(int(permutationTmp[m]))][str(m)] = list()
                                senOri[str(int(permutationTmp[m]))][str(m)].append(Attn_States[i+j, -1, batch.data['Index_Sample'][i+j][m][0]:batch.data['Index_Sample'][i+j][m][1],:].reshape(-1, HiddenSize))
                    
                    for j in range(numPerSam):
                        permutationTmp = batch_gold.data['permutation'][i+j]
                        senGold[str(int(permutationTmp[-2]))] = Attn_States_gold[i+j, -1, batch_gold.data['Index_Sample'][i+j][-2][0]:batch_gold.data['Index_Sample'][i+j][-2][1],:].reshape(-1, HiddenSize)

                    Temperature = 0.1
                    cos = nn.CosineSimilarity(dim=1)
                    num = 0
                    for j in senOri.keys():
                        if senGold[j] != {}:
                            for m in senOri[j].keys():
                                for k in range(len(senOri[j][m])):
                                    query = senOri[j][m][k]
                                    
                                    key = senGold[j]
                                    multi = cos(query, key)
                                    multi /= Temperature
                                    positive = torch.exp(multi)
                                    
                                    multiNeg = cos(query, query.clone().detach())
                                    multiNeg /= Temperature
                                    negative = torch.exp(multiNeg)
                                    
                                    loss_tmp = - torch.log(positive / (positive + negative))
                                    loss_tmp = torch.mean(loss_tmp)
                                    loss_contrast += loss_tmp
                                    num += 1

                     
                    Loss_Consistency = 0
                    Hidden1 = list()
                    Hidden2 = list()
                    Hiddenstates = list()
                    for i in range(0, batch_size):
                        Hiddenstates.append(output.Hidden_states[i][batch.data['prompt_len'][i] - batch.data['label_len'][i] - 1])
                        
                    gold = torch.stack(Hiddenstates, dim=0).mean(dim=0)
                    for m in range(numPerSam):
                        Hidden1.append(Hiddenstates[m])
                        Hidden2.append(gold)
                    Hidden1 = torch.cat(Hidden1, dim=0)
                    Hidden2 = torch.cat(Hidden2, dim=0)
                    Target = torch.ones(Hidden1.shape[0]).to(Hidden1.device)
                    Hidden2 = Hidden2.to(Hidden1.device)
                    loss_fn_consistency = torch.nn.CosineEmbeddingLoss()
                    Loss_Consistency = loss_fn_consistency(Hidden1, Hidden2, Target)
                    Loss_Consistency = Loss_Consistency.to(loss_contrast.device)
                
                loss_contrast /= num

                
                loss_main = output.loss

                loss = loss_contrast + Loss_Consistency


                loss = loss / gradient_accumulation_steps
                if total_loss != 0:
                    total_loss = total_loss.to(loss.device)
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(gold_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update(step//gradient_accumulation_steps)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(gold_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(step//gradient_accumulation_steps)
                
                # pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(gold_dataloader)} completed (loss_main: {loss_main.detach().float()}) loss_constrast: {loss_contrast.detach().float()}")
                pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(gold_dataloader)} completed (loss_main: {loss_main.detach().float()}) loss_constrast: {loss_contrast.detach().float()} Loss_Consistency: {Loss_Consistency.detach().float()}")
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)    
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(gold_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        print(f"Max CUDA memory allocated was {memtrace.peak} GB")
        print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
        print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        
        # Update the learning rate as needed
        lr_scheduler.step()

        # save the model
        model.save_pretrained(train_config.output_dir, save_adapter=True, save_config=True)
        train_config.run_validation = False
        
        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
    

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    # Some user added configs

    num_sample = int(kwargs['NumTrain'])
    
    if kwargs['Dataset'] == 'SST5':
        train_config.output_dir = 'savedmodel/SST-5/SST5_' + str(num_sample) + '_'+kwargs['Model'] + '_Lora8_InfoAC'
    elif kwargs['Dataset'] == 'SST2':
        train_config.output_dir = 'savedmodel/SST-2/SST2_' + str(num_sample) + '_'+kwargs['Model'] + '_Lora8_InfoAC'
    elif kwargs['Dataset'] == 'Round':
        train_config.output_dir = 'savedmodel/Round/Round_' + str(num_sample) + '_'+kwargs['Model'] + '_Lora8_InfoAC'
    elif kwargs['Dataset'] == 'Next':
        train_config.output_dir = 'savedmodel/Next/Next_' + str(num_sample) + '_'+kwargs['Model'] + '_Lora8_InfoAC'
    elif kwargs['Dataset'] == 'qqp':
        train_config.output_dir = 'savedmodel/qqp/qqp_' + str(num_sample) + '_'+kwargs['Model'] + '_Lora8_InfoAC'
    
    print(train_config.output_dir)

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    # Load the pre-trained model and setup its configuration
    
    
    model, model_gold, tokenizer = load_model(kwargs, train_config)

    print_model_size(model, train_config, 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        peft_config.target_modules = ["q_proj", "v_proj"]
        peft_config.r = 8
        peft_config.lora_alpha = 16
        print(peft_config.r)
        print(peft_config.target_modules)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # SST-5
    if kwargs['Model'] == "LLama-7B":
        model_dataset = "LLama"
    elif kwargs['Model'] == "LLama-13B":
        model_dataset = "LLama"
    elif kwargs['Model'] == "Vicuna-7B" or kwargs['Model'] == "Vicuna-13B":
        model_dataset = "Vicuna"
    else:
        model_dataset = kwargs['Model']
    
    if kwargs['Dataset'] == 'SST5':
        OutputTrain = "MyDataset/SST-5/SST5-" + model_dataset + "-Pool100-Len10-Train.pickle"
        OutputGoldTraincase = "MyDataset/SST-5/SST5-" + model_dataset + "-Pool100-Len10-Gold.pickle"
    elif kwargs['Dataset'] == 'SST2':
        OutputTrain = "MyDataset/sst-2/SST2-" + model_dataset + "-Pool100-Len10-Train.pickle"
        OutputGoldTraincase = "MyDataset/sst-2/SST2-" + model_dataset + "-Pool100-Len10-Gold.pickle"
    elif kwargs['Dataset'] == 'Round':
        OutputTrain = "MyDataset/Round/Round-" + model_dataset + "-Pool10-Len10-Train.pickle"
        OutputGoldTraincase = "MyDataset/Round/Round-" + model_dataset + "-Pool10-Len10-Gold.pickle"
    elif kwargs['Dataset'] == 'Next':
        OutputTrain = "MyDataset/Next/Next-" + model_dataset + "-Pool10-Len10-Train.pickle"
        OutputGoldTraincase = "MyDataset/Next/Next-" + model_dataset + "-Pool10-Len10-Gold.pickle"
    elif kwargs['Dataset'] == 'qqp':
        OutputTrain = "MyDataset/qqp/qqp-" + model_dataset + "-Pool100-Len10-Train.pickle"
        OutputGoldTraincase = "MyDataset/qqp/qqp-" + model_dataset + "-Pool100-Len10-Gold.pickle"
    

    # OutputDevList = OutputTrain
    Num = num_sample*train_config.batch_size_training
    dataset_train = InContextDataTrain(OutputTrain, Num, partition="train")
    
    dataset_gold = InContextDataTrain(OutputGoldTraincase, Num, partition="train")

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "val")
    train_dl_kwargs['batch_sampler'].shuffle = False
    train_dl_kwargs['batch_sampler'].batch_size = train_config.batch_size_training
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    gold_dl_kwargs = get_dataloader_kwargs(train_config, dataset_gold, tokenizer, "val")
    gold_dl_kwargs['batch_sampler'].shuffle = False
    gold_dl_kwargs['batch_sampler'].batch_size = train_config.batch_size_training
    gold_dataloader = torch.utils.data.DataLoader(
        dataset_gold,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **gold_dl_kwargs,
    )

    # Initialize the optimizer and learning rate scheduler
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    train(
        model,
        model_gold,
        train_dataloader,
        gold_dataloader,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config
    )
    

if __name__ == "__main__":

    fire.Fire(main)

    
    