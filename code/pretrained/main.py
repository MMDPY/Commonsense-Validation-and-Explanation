####################################################
# CMPUT 501
# Commonsense Validation and Explanation
# karimiab & azamani1
# Task 1 using GPT-2 & Bert LM and Bert classifier
####################################################

# Prerequisites
import os
import time
import datetime
import random
import sys
import torch
import math
import numpy as np
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
from preprocessing import data_loader, classification_data
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertLMHeadModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_bert import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# constant variables
train_d_path = '../../data/Training_Data/subtaskA_data_all.csv'
train_a_path = '../../data/Training_Data/subtaskA_answers_all.csv'
dev_d_path = '../../data/Dev_Data/subtaskA_dev_data.csv'
dev_a_path = '../../data/Dev_Data/subtaskA_gold_answers.csv'
test_d_path = '../../data/Test_Data/subtaskA_test_data.csv'
test_a_path = '../../data/Test_Data/subtaskA_gold_answers.csv'
output_path = '../../runs/'


def format_time(delta):
    """
    Converting the elapsed time into a human-readable format
    Parameters:
        delta
    Returns:
        string representation of the input passed time
    """
    delta = int(round(delta))
    return str(datetime.timedelta(seconds=delta))


def calc_pp(sentence, model, tokenizer):
    """
    Calculates the perplexity of a given sentence
    Parameters:
        sentence (str): input sentence
        model ():
        tokenizer ():
    Returns:
        Perplexity of the input sentence
    """
    input_vector = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_vector = input_vector.to('cpu')
    with torch.no_grad():
        outputs = model(input_vector, labels=input_vector)
    loss = outputs[0]
    return math.exp(loss)


def save_results(df, path):
    """
        Saves the result into a .csv file
        Parameters:
            df (str): input sentence
            path ():

        Returns:
            N/A
        """
    if path:
        df.to_csv(path)


def gpt():
    """
    Loads the pretrained GPT2 language model and the dataset to estimate
    the probability (inverse perplexity) of the given sentences
    """
    version = 'distilgpt2'
    model = GPT2LMHeadModel.from_pretrained(version)
    tokenizer = GPT2Tokenizer.from_pretrained(version)
    test_df = data_loader(test_d_path, test_a_path)
    N = test_df.shape[0]
    for j, row in tqdm(test_df.iterrows()):
        first_sentence = row['sentence1']
        second_sentence = row['sentence2']
        first_pp = calc_pp(first_sentence, model, tokenizer)
        second_pp = calc_pp(second_sentence, model, tokenizer)
        test_df.loc[j, 'PP'] = first_pp
        test_df.loc[j, 'PP'] = second_pp
        test_df.loc[j, 'not_making_sense'] = np.argmax([first_pp, second_pp]).astype('int64')

    save_results(test_df, output_path + 'gptLM.csv')
    print('GPT-2 accuracy on the test set is equal to {:.2f}'.format(((test_df['label'] == test_df['not_making_sense']).sum() / N) * 100))


def bert():
    """
    Loads the pretrained BERT language model and the dataset to estimate
    the probability (inverse perplexity) of the given sentences
    """

    version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(version)
    model = BertLMHeadModel.from_pretrained(version, is_decoder=True)
    test_df = data_loader(test_d_path, test_a_path)
    N = test_df.shape[0]
    for i, row in tqdm(test_df.iterrows()):
        first_sentence = row['sentence1']
        second_sentence = row['sentence2']
        first_pp = calc_pp(first_sentence, model, tokenizer)
        second_pp = calc_pp(second_sentence, model, tokenizer)
        test_df.loc[i, 'PP'] = first_pp
        test_df.loc[i, 'PP'] = second_pp
        test_df.loc[i, 'not_making_sense'] = np.argmax([first_pp, second_pp]).astype('int64')

    save_results(test_df, output_path + 'bertLM.csv')
    print('Bert accuracy on the test set is equal to {:.2f}'.format(((test_df['label'] == test_df['not_making_sense']).sum() / N) * 100))


def accuracy_list(prediction, ground_truth):
    """
    Calculates the accuracy of the input prediction and ground truth labels
    Parameters:
        ground_truth ():
        prediction ():
    Returns:
        The calculated accuracy w.r.t the input parameters

    """
    pred_flat = np.argmax(prediction, axis=1).flatten()
    golden_labels = ground_truth.flatten()
    return np.sum(pred_flat == golden_labels) / len(golden_labels)


def load_model():
    """
    Loads the saved weights by using from_pretrained module and evaluates the model
    Parameters:
        N/A
    Returns:
        ground_truth ():
        predictions ():
        """
    device = 'cpu'
    output_dir = '../checkpoints/'
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    df = classification_data('test')
    print('Number of test sentences is equal to:', df.shape[0])
    ground_truth, predictions = evaluation(df, model, tokenizer, device)

    print('The model\'s accuracy on the test set is equal to:',
          np.sum(predictions == ground_truth)/len(ground_truth), '\n')
    print(confusion_matrix(ground_truth, predictions))
    print(classification_report(ground_truth, predictions,
                                target_names=['Makes Sense', 'Doesn\'t Make Sense']))

    kf = KFold(n_splits=5)
    accuracy_score = []
    f1_score = []
    for train_index, test_index in kf.split(ground_truth):
        ground_truth_split = ground_truth[test_index]
        predictions_split = predictions[test_index]
        accuracy_score.append(np.sum(predictions_split == ground_truth_split)/len(ground_truth_split))
        f1_score.append(classification_report(ground_truth_split, predictions_split, output_dict=True)['macro avg']['f1-score'])
    print('mean accuracy is equal to: ', np.mean(accuracy_score))
    print('mean f1-score is equal to: ', np.mean(f1_score))

    return ground_truth, predictions


def evaluation(df, model, tokenizer, device):
    """
    Evaluates the models using the input dataframe and parameters
    Parameters:
        df (dataframe):
        model (object):
        tokenizer (object):
        device(str):
    Returns:
        N/A
    """
    # extracting the sentences from the input dataframe
    sentences = df.sentence.values
    # extracting the labels from the input dataframe
    labels = df.label.values
    # creating token id list for tokenization purposes
    input_ids = []
    # defining attention masks list for tokenization purposes
    attention_masks = []

    # tokenization procedure
    for sent in sentences:
        encoding = tokenizer.encode_plus(sent, max_length=64, truncation=True,
                                         padding="max_length", add_special_tokens=True,
                                         return_attention_mask=True, return_tensors='pt')

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    batch_size = 32
    # transforming the token id, masks, and labels list to pytorch tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # constructing a tensor dataset using the input data
    test_data = TensorDataset(input_ids, attention_masks, labels)
    test_sampler = SequentialSampler(test_data)
    test_dl = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    print('Starting the evaluation procedure for {} test sentences!'.format(len(input_ids)))

    model.eval()

    ground_truth, predictions = [], []
    eval_acc = 0

    for batch in test_dl:
        batch = tuple(t.to(device) for t in batch)
        # extracting inputs from the batch
        b_input_ids, b_token_mask, b_labels = batch
        # while ignoring gradient computation
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_token_mask)

        # storing the output into logtis variable
        logits = outputs[0]
        # converting logits and labels into cpu format
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        ground_truth.append(label_ids)
        predictions.append(logits)

        eval_acc += accuracy_list(logits, label_ids)

    avg_val_accuracy = eval_acc / len(test_dl)
    print('Accuracy is equal to: ', avg_val_accuracy)
    print('Calculation finished!')

    # taking the argmax, since we want the label with the maximum accuracy
    predictions = np.argmax(np.concatenate(predictions, axis=0), axis=1).flatten()
    # converting all ground_truth values across batches into list
    ground_truth = np.concatenate(ground_truth, axis=0)

    return ground_truth, predictions


def tune_bert():
    # using cpu for training
    device = 'cpu'
    # specifying the bert version
    version = 'bert-base-uncased'
    # importing the tokenizer with the corresponding version
    tokenizer = BertTokenizer.from_pretrained(version, do_lower_case=True)

    # df = data_loader(train_d_path, train_a_path)
    df = classification_data('train')
    sentences = df.sentence.values
    labels = df.label.values
    # print(len(sentences), len(labels))
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoding = tokenizer.encode_plus(sent, max_length=64, truncation=True,
                                         padding="max_length", add_special_tokens=True,
                                         return_attention_mask=True, return_tensors='pt'
                                         )

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    # print(len(input_ids))
    # converting the input lists to torch tensors
    batch_size = 32
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # constructing the tensor dataset using TD class
    data = TensorDataset(input_ids, attention_masks, labels)

    # 90 to 10 split for training and development sets
    train_split = int(len(data) * 0.9)
    validation_split = len(data) - train_split

    # randomly shuffling and selecting the samples
    train_data, val_data = random_split(data, [train_split, validation_split])

    print('Number of training samples are equal to:', train_split)
    print('Number of validation samples are equal to:', validation_split)

    # The random sampler selects the batches randomly
    train_dl = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    # The sequential sampler chooses the batches sequentially
    validation_dl = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

    # initializing the binary classification BERT model
    model = BertForSequenceClassification.from_pretrained(version,
                                                          num_labels=2, output_attentions=False,
                                                          output_hidden_states=False)
    # using AdamW as the models optimizer
    # Adam is a stochastic optimization method that modifies the typical implementation of weight decay
    optimizer = AdamW(model.parameters(), eps=1e-8, lr=5e-5)
    epochs = 2
    steps = epochs * len(train_dl)

    # learning rate scheduler to update the parameters
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)
    # torch.cuda.manual_seed_all(seed_val)
    seed_val = 27
    random.seed(seed_val)
    torch.manual_seed(seed_val)

    # saving the current time
    t0 = time.time()
    logging = []

    for epoch in range(epochs):
        print('Starting to Train!')
        print('%%%%%%%%% Epoch {} from {} %%%%%%%%%'.format(epoch + 1, epochs))

        training_loss = 0
        ti = time.time()
        model.train()
        for step, batch in enumerate(train_dl):
            # logging after every two batches
            if step % 64 == 0 and not step == 0:
                elapsed = format_time(time.time() - ti)
                print('Batch {} of {:>5,}. Time elapsed: {}.'.format(step, len(train_dl), elapsed))

            # extracting pytorch tensors
            # first batch contains the token ids
            b_input_ids = batch[0].to(device)
            # second batch contains the masks
            b_input_mask = batch[1].to(device)
            # third batch contains the labels
            b_labels = batch[2].to(device)

            # vanishing the gradient for backpropagation purpose
            model.zero_grad()

            loss, logits = model(b_input_ids, token_type_ids=None,
                                 attention_mask=b_input_mask, labels=b_labels)
            # print(loss)
            training_loss += loss.item()

            # applying bpp
            loss.backward()

            # gradient clipping (normalization)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # updating the parameters and learning rate
            optimizer.step()
            scheduler.step()

            print("So far, training took {} (hour:min:seconds)".format(format_time(time.time()-ti)))

            optimizer.step()
            scheduler.step()

        # averaging the loss over all batches
        average_train_loss = training_loss / len(train_dl)
        training_time = format_time(time.time() - ti)

        print('=' * 25)
        print('Average training loss is equal to:', average_train_loss)
        print('Training epoch took: {}'.format(training_time))
        print('='*25)

        print("Validation in process!")

        ti = time.time()
        model.eval()

        total_loss = 0
        total_accuracy = 0

        for batch in validation_dl:
            # extracting batch objects
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                loss, logits = model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask, labels=b_labels)
            total_loss += loss.item()
            # transferring logits and label ids to cpu numpy array
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_accuracy += accuracy_list(logits, label_ids)

        avg_val_acc = total_accuracy / len(validation_dl)
        avg_val_loss = total_loss / len(validation_dl)
        val_time = format_time(time.time() - ti)
        print("Accuracy is equal to: {}".format(avg_val_acc))
        print("Validation loss is equal to: {}".format(avg_val_loss))
        print("Validation took: {}".format(val_time))

        logging.append(
            {
                'epoch': epoch + 1,
                'Training Time': training_time,
                'Validation Time': val_time,
                'Validation Accuracy.': avg_val_acc,
                'Training Loss': average_train_loss,
                'Validation Loss': avg_val_loss
            }
        )

    print('='*25)
    print("Training complete!")
    print("The training process took {} (hour:min:seconds)".format(format_time(time.time()-t0)))

    # Testing phase
    df = classification_data('test')

    print('Number of test sentences is equal to:', df.shape[0])
    sentences = df.sentence.values
    labels = df.label.values

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoding = tokenizer.encode_plus(sent, max_length=64, truncation=True,
                                         padding="max_length", add_special_tokens=True,
                                         return_attention_mask=True, return_tensors='pt'
                                         )

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    # converting the token ids, masks, and labels to pytorch tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    batch_size = 32
    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    test_sampler = SequentialSampler(test_dataset)
    test_dl = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    print('Predicting the output label for {} test sentences!'.format(len(input_ids)))
    # evaluating on the test set
    model.eval()

    ground_truth, predictions = [], []
    eval_acc = 0

    for batch in test_dl:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        ground_truth.append(label_ids)
        predictions.append(logits)
        eval_acc += accuracy_list(logits, label_ids)

    avg_val_accuracy = eval_acc / len(test_dl)
    print("Accuracy is equal to: {}".format(avg_val_accuracy))
    print('Evaluation finished!')

    output_dir = '../checkpoints/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # check if the model has module attr
    if hasattr(model, 'module'):
        result = model.module
    else:
        result = model

    result.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    model = sys.argv[1]

    if model == 'gpt':
        gpt()
    elif model == 'bert':
        bert()
    elif model == 'tuned_bert':
        tune_bert()
    elif model == 'load':
        load_model()
    else:
        print('You entered an incorrect input argument, please try again!')


if __name__ == "__main__":
    main()