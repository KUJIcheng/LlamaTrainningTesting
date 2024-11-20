from torch.utils.data import Dataset

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["text"], padding='max_length', max_length=max_length, truncation=True)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text on-the-fly
        encoding = tokenize_function(self.texts[idx], self.tokenizer, self.max_length)
        
        return {
            "input_ids": encoding['input_ids'],
            "labels": encoding['input_ids'],
            "attention_mask": encoding['attention_mask'],
        }