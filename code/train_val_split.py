from sklearn.model_selection import train_test_split
from tokenize_data import encode_text, encode_labels
from create_tensor_dataset import ToxicSpansDetection

def train_val_split(tokenized_texts, labels, test_size):
    print('> Creating training and validation datasets..\n')

    train_texts, val_texts, train_labels, val_labels = train_test_split(tokenized_texts, labels, test_size=test_size)
    
    train_encodings = encode_text(train_texts)
    val_encodings = encode_text(val_texts)

    train_encoded_labels = encode_labels(train_labels, train_encodings)
    val_encoded_labels = encode_labels(val_labels, val_encodings)

    # we don't want to pass this to the model
    train_encodings.pop("offset_mapping") 
    val_encodings.pop("offset_mapping")

    train_dataset = ToxicSpansDetection(train_encodings, train_encoded_labels)
    val_dataset = ToxicSpansDetection(val_encodings, val_encoded_labels)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print('-- Done!\n')

    return train_dataset, val_dataset
    
