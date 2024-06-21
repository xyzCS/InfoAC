# Addressing Order Sensitivity of In-Context Demonstration Examples in Causal Language Models. (InfoAC)

Code release for [Addressing Order Sensitivity of In-Context Demonstration Examples in Causal Language Models](https://arxiv.org/abs/2402.15637.pdf). [Findings of ACL 2024]

## Installation
```bash
conda create -n infoac python=3.10
conda activate infoac
pip install -r requirements.txt
```

## File Organization
```
InfoAC
|-- MyDataset                                            # Contains the processed data files.
|   |-- SST-5                                            # DataFiles for the SST-5 benchmark.
    |   |-- SST5-LLama-Pool100-Len10-Test.pickle         # Test set.
    |   |-- SST5-LLama-Pool100-Len10-Train.pickle        # Training set.
    |   |-- SST5-LLama-Pool100-Len10-Gold.pickle         # Reference set for the reference model.
|-- savedmodel                                           # Saved checkpoints after fine-tuning with InfoAC.
    |   |-- SST-5                                        # Saved checkpoints for the SST-5 benchmark.
        |-- SST5_1000_LLama-7B_Lora8_InfoAC              # Checkpoints for the LLama-7B.                                     


```

## Evaluation
The experiments utilize four large language models (LLMs): LLama2-7B-chat, LLama2-13B-chat, Vicuna-7B-v1.5, and Vicuna-13B-v1.5.
```
**Original LLM**
python Evaluation.py  --quantization --Model "LLama-7B" --Dataset "SST5"
Model = ['LLama-7B', 'LLama-13B', 'Vicuna-7B', 'Vicuna-13B']
Dataset = ['SST5', 'SST2', 'Next', 'Round', 'QQP']

LLM after Fine-tuning with InfoAC:
python Evaluation.py --quantization  --load_path='savedmodel/SST-5/SST5_1000_LLama-7B_Lora8_InfoAC' --Model "LLama-7B" --Dataset "SST5"




```

## fine-tuning
```

```


## Reference
```
@article{Xiang2024AddressingOS,
  title={Addressing Order Sensitivity of In-Context Demonstration Examples in Causal Language Models},
  author={Yanzheng Xiang and Hanqi Yan and Lin Gui and Yulan He},
  journal={ArXiv},
  year={2024},
  volume={abs/2402.15637},
  url={https://api.semanticscholar.org/CorpusID:267938656}
}
```
