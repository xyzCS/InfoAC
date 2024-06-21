import copy
import fire
import random
from peft import PeftModel
import torch
import numpy as np
from peft import prepare_model_for_int8_training

from transformers import (
    LlamaTokenizer,
)
from LLama import LlamaForCausalLM
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from trainingconfig import train_config as TRAIN_CONFIG
from llama_recipes.utils.config_utils import (
    update_config,
)
from transformers import LlamaTokenizer
import pickle


def loadmodel(kwargs):
    if kwargs['Model'] == "LLama-7B":
        modelpath = 'meta-llama/Llama-2-7b-chat-hf'
        tokenizer = LlamaTokenizer.from_pretrained(
            modelpath,
            use_fast=False,
            padding_side="left",
            )
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        model = LlamaForCausalLM.from_pretrained(
            modelpath, 
            load_in_8bit=True,
            device_map="auto",
            )
    elif kwargs['Model'] == "LLama-13B":
        modelpath = 'meta-llama/Llama-2-13b-chat-hf'
        tokenizer = LlamaTokenizer.from_pretrained(
            modelpath,
            use_fast=False,
            padding_side="left",
            device_map="auto",
            )
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        model = LlamaForCausalLM.from_pretrained(
            modelpath, 
            load_in_8bit=True,
            device_map="auto",
            )
    elif kwargs['Model'] == "Vicuna-7B":
            modelpath = 'lmsys/vicuna-7b-v1.5'
            tokenizer = LlamaTokenizer.from_pretrained(
                modelpath,
                use_fast=False,
                padding_side="left",
                device_map="auto",
            )
            tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            tokenizer.bos_token_id = 1
            model = LlamaForCausalLM.from_pretrained(
                modelpath, 
                load_in_8bit=True,
                device_map="auto",
            )
    elif kwargs['Model'] == "Vicuna-13B":
        modelpath = 'lmsys/vicuna-13b-v1.5'
        tokenizer = LlamaTokenizer.from_pretrained(
            modelpath,
            use_fast=False,
            padding_side="left",
            device_map="auto",
        )
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        model = LlamaForCausalLM.from_pretrained(
            modelpath, 
            load_in_8bit=True,
            device_map="auto",
        )
    return model, tokenizer


def EvaluateTheOutputOfFinetuning(OutputData):
    Numcorrect = 0
    NumALLCorrect = 0
    NumPartialCorrect = 0
    Entropy_Metric_goldall = list()
    i = 0
    Num = 20
    while i < len(OutputData):
        answer = dict()
        for j in range(Num):
            if OutputData[i+j]['output'].lower().strip() == 'not\\_entailment' or OutputData[i+j]['output'].lower().strip() == 'not\_entailment':
                OutputData[i+j]['output'] = 'not_entailment'
            if OutputData[i+j]['output'].lower().strip() == 'not\\_duplicate' or OutputData[i+j]['output'].lower().strip() == 'not\_duplicate':
                OutputData[i+j]['output'] = 'not_duplicate'
            if OutputData[i+j]['output'].lower().strip() not in answer.keys():
                answer[OutputData[i+j]['output'].lower().strip()] = 1
            else:
                answer[OutputData[i+j]['output'].lower().strip()] += 1

        answer2 = copy.deepcopy(answer)
        answer = sorted(answer.items(), key=lambda x:x[1], reverse=True)
        if answer[0][0].lower().strip() == 'not\\_entailment' or answer[0][0].lower().strip() == 'not\_entailment':
            answer[0] = ('not_entailment', answer[0][1])

        if answer[0][0].lower().strip() == 'not\\_duplicate' or answer[0][0].lower().strip() == 'not\_duplicate':
            answer[0] = ('not_duplicate', answer[0][1])

        if answer[0][0].lower().strip() == OutputData[i+j]['label'].lower().strip():
            distribution = list()
            for n in answer2.keys():
                distribution.append(answer2[n]/Num)
            distribution = np.array(distribution)
            log_distribution = np.log2(distribution)
            shang = -1 * np.sum(distribution * log_distribution, axis=0)
            Entropy_Metric_goldall.append(shang)
            Numcorrect += 1
            if answer[0][1] == Num:
                NumALLCorrect += 1
            else:
                NumPartialCorrect += 1  
        for j in range(Num):  
            OutputData[i+j]['MajorPredict'] = answer[0][0].lower().strip()
            OutputData[i+j]['label'] = answer[0][0].lower().strip()
        i += Num

    All = len(OutputData) / Num
    Entropy = np.mean(Entropy_Metric_goldall)
    print("Accuracy:{}".format(Numcorrect/All)) 
    print("Partial Correct Ratio:{}".format(NumPartialCorrect/Numcorrect))  
    print("Entropy:{}".format(Entropy))


def evaluate(model, tokenizer, ModelName, dataset):
    if ModelName == "LLama-7B" or ModelName == "LLama-13B":
        ModelName = "LLama"
    elif ModelName == "Vicuna-7B" or ModelName == "Vicuna-13B":
        ModelName = "Vicuna"

    if dataset == "SST5":
        datapath = "MyDataset/SST-5/SST5-" + ModelName + "-Pool100-Len10-Test.pickle"
    elif dataset == "SST2":
        datapath = "MyDataset/sst-2/SST2-" + ModelName + "-Pool100-Len10-Test.pickle"
    elif dataset == "Round":
        datapath = "MyDataset/Round/Round-" + ModelName + "-Pool10-Len10-Test.pickle"
    elif dataset == "Next":
        datapath = "MyDataset/Next/Next-" + ModelName + "-Pool10-Len10-Test.pickle"
    elif dataset == "qqp":
        datapath = "MyDataset/qqp/qqp-" + ModelName + "-Pool100-Len10-Test.pickle"
    
    Evaluatedata = pickle.load(open(datapath, 'rb'))
    model.eval()
    batch_size = 2
    model.generation_config.do_sample = False
    with torch.no_grad():
        NumCorrect = 0
        i = 0
        while i < len(Evaluatedata):
            print(i)
            sentence = list()
            input_length = list()
            for k in range(batch_size):
                prompt = Evaluatedata[i+k]['prompt']
                sentence.append(prompt)
                model_input = tokenizer(prompt, return_tensors="pt")
                input_length.append(model_input['input_ids'].shape[-1])
            model_input = tokenizer(sentence, return_tensors="pt", padding=True).to("cuda")
            output_sequences = model.generate(**model_input, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            output_sentence = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
            for k in range(output_sequences.shape[0]):
                outputtmp = output_sentence[k][len(sentence[k]):]
                indexStop = outputtmp.find('\n', 0)
                if indexStop != -1:
                    outputtmp = outputtmp[0:indexStop]
                output = outputtmp.lower().strip()
                print(output)
                Evaluatedata[i+k]['output'] = output
                if output.lower().strip() == Evaluatedata[i+k]["label"].lower().strip():
                    NumCorrect += 1
                Evaluatedata[i+k]['Correct'] = output.lower().strip() == Evaluatedata[i+k]["label"].lower().strip()
            
            i += batch_size
    EvaluateTheOutputOfFinetuning(Evaluatedata)


def Inference_Lora_finetuned_LLama(kwargs):
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
     
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    # Load Model
    model, tokenizer = loadmodel(kwargs)
    
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)
    
    if 'load_path' in kwargs.keys():
        model = PeftModel.from_pretrained(model, kwargs['load_path'])

    evaluate(model, tokenizer, kwargs['Model'], kwargs['Dataset'] )


def Evalute(**kwargs):
    if kwargs['Model'] == "LLama-7B":
        Inference_Lora_finetuned_LLama(kwargs)
    elif kwargs['Model'] == "LLama-13B":
        Inference_Lora_finetuned_LLama(kwargs)
    elif kwargs['Model'] == "Vicuna-7B":
        Inference_Lora_finetuned_LLama(kwargs)
    elif kwargs['Model'] == "Vicuna-13B":
        Inference_Lora_finetuned_LLama(kwargs)


if __name__ == "__main__":
    fire.Fire(Evalute)
    
 
    