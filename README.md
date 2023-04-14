# Apostrophe/Quote Prediction using Transformers

This is an implementation of Transformers and LSTM to solve the problem of quotation prediction. The trained model is able to guess in which positions a single or double quote should be put. LSTM model is trained from scratch and character-based. Addition to LSTM, BERT and T5 models are used. According to the experiments, T5 seems to outperform other models. Refer to the [report](./report.pdf) for implementation details and results.

## How to Run

### Data Fetching/Preprocessing

```bash
python data.py --min-len 50 --max-len 500 --silicone --wiki --output-path ./dataset/

optional arguments:
  -h, --help            show this help message and exit
  --min-len MIN_LEN     Discard sentences with length less than this value
  --max-len MAX_LEN     Discard sentences with length greater than this value
  --silicone            Include informal datasets
  --wiki                Include formal dataset
  --output-path OUTPUT_PATH
                        Output path for gathered data
```

### Training/Evaluation

Please check 'config.py' for all configuration options. They should be self-explanatory. Also, there are example configurations for all three of
models in configs/ folder.

For example, to train a BERT model: 

```bash
python main.py --config configs/bert.yml
```