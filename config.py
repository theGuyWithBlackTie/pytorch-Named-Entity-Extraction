DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN          = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS           = 10

BERT_MODEL_PATH  = "bert-base-uncased"
MODEL_SAVE_PATH  = "../output/"
DATA_FILE        = "data/ner_dataset.csv"
TOKENIZER        = transformers.BertTokenizer.from_pretrained(BERT_MODEL_PATH, do_lower_case=True)