# Offensive Language Identification for Dravidian languages
This code is for the "Syntax-aware Offensive Content Identification in Low-resourced Code-mixed Languages with Continual Pre-training" paper.

You can use the model with the parameters in the train.json file.
Syntax-BERT model is adopted from["Improving BERT with Syntax-aware Local Attention"](https://aclanthology.org/2021.findings-acl.57.pdf)

## Usage
```
python main.py train.json
```
**model options** (code in train.json)
- bert: BERT
- cont-BERT: continual training BERT
- syntax_bert: Syntax-BERT

To train Cont-Syntax-BERT, you need to run cont-BERT than run syntax-BERT with the trained model of cont-BERT.

## Results

Results on DravidianCodeMix dataset:

### Tamil

| Model            	| Precision 	| Recall 	| F1-Score 	| 
|------------------	|-----------	|--------	|----------	|
| BERT             	| 0.53      	| 0.73   	| 0.61     	|
| Syntax-BERT      	| 0.55      	| 0.73   	| 0.62     	|
| Cont-BERT        	| 0.75      	| 0.77   	| 0.76     	|
| Cont-Syntax-BERT 	| 0.84      	| 0.76   	| 0.80     	|

### Kannada

| Model            	| Precision 	| Recall 	| F1-Score 	| 
|------------------	|-----------	|--------	|----------	|
| BERT             	| 0.50      	| 0.64   	| 0.56     	|
| Syntax-BERT      	| 0.71      	| 0.76   	| 0.73     	|
| Cont-BERT        	| 0.72      	| 0.74   	| 0.73     	|
| Cont-Syntax-BERT 	| 0.77      	| 0.76   	| 0.76     	|

### Malayalam

| Model            	| Precision 	| Recall 	| F1-Score 	| 
|------------------	|-----------	|--------	|----------	|
| BERT             	| 0.83      	| 0.88   	| 0.78     	|
| Syntax-BERT      	| 0.87      	| 0.90   	| 0.87     	|
| Cont-BERT        	| 0.94      	| 0.95   	| 0.94     	|
| Cont-Syntax-BERT 	| 0.96      	| 0.97   	| 0.96     	|
