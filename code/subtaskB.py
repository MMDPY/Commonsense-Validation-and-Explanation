import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaTokenizer

import os
import time
import datetime
import numpy as np

from torch.utils.data import Dataset
import csv

class DatasetB(Dataset):

    def __init__(self, root='./data/Training_Data/', maxlen=64):

        print('Read data from: ', root)
        label = open(root+'/subtaskB_answers.csv')
        data = open(root+'/subtaskB_data.csv')
        self.labels = []
        self.data = []
        option_to_label = {'A':0, 'B':1, 'C':2}
        
        reader_label = csv.reader(label)
        for row in reader_label:
            id = int(row[0])
            label = int(option_to_label[row[1]])
            self.labels.append((id, label))

        reader_data = csv.reader(data)  
        next(reader_data, None)  # skip the headers  
        for row in reader_data:
            id = int(row[0])
            sent = str(row[1])
            option_A = str(row[2])
            option_B = str(row[3])
            option_C = str(row[4])
            
            if sent[-1] == '.':
              sent = sent[:-1]
            if option_A != '' and option_A[-1] == '.':
              option_A = option_A[:-1]
            if option_B != '' and option_B[-1] == '.':
              option_B = option_B[:-1]
            if option_C != '' and option_C[-1] == '.':
              option_C = option_C[:-1]
          
            self.data.append((id, sent, option_A, option_B, option_C))

        #Initialize the BERT tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

        self.maxlen = maxlen

    def prepare_features(self, sent, tokenizer, max_seq_length = 64, 
             zero_pad = True, include_CLS_token = True, include_SEP_token = True):
      ## Tokenzine Input
      tokens_a = tokenizer.tokenize(sent)

      ## Truncate
      if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[0:(max_seq_length - 2)]
      ## Initialize Tokens
      tokens = []
      if include_CLS_token:
          tokens.append(tokenizer.cls_token)
      ## Add Tokens and separators
      for token in tokens_a:
          tokens.append(token)

      if include_SEP_token:
          tokens.append(tokenizer.sep_token)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      ## Input Mask 
      input_mask = [1] * len(input_ids)
      ## Zero-pad sequence lenght
      if zero_pad:
          while len(input_ids) < max_seq_length:
              input_ids.append(0)
              input_mask.append(0)
      return torch.tensor(input_ids), torch.tensor(input_mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        id = self.data[index][0]
        sent = self.data[index][1]
        option_A = self.data[index][2]
        option_B = self.data[index][3]
        option_C = self.data[index][4]
        
        assert id == self.labels[index][0]
        label = self.labels[index][1]
        label = torch.eye(3)[label]

        tesor_1, mask_1 = self.prepare_features(sent+option_A, self.tokenizer) # Tokenize the sentence
        tensor_2, mask_2 = self.prepare_features(sent+option_B, self.tokenizer) # Tokenize the sentence
        tensor_3, mask_3 = self.prepare_features(sent+option_C, self.tokenizer) # Tokenize the sentence
        
        return tesor_1, tensor_2, tensor_3, mask_1, mask_2, mask_3, label, id
    
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class RobertaClassifier(nn.Module):
    def __init__(self, cnt_labels=1, dropout_prob=0.15, freeze = False):
        super(RobertaClassifier, self).__init__()
        self.layer = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout_prob)
        if freeze:
            for param in self.layer.parameters():
                param.requires_grad = False
        # For calculation of the score of a sentence
        self.classifier = nn.Linear(768, cnt_labels)
        
    def forward(self, tensor_1, tensor_2, tensor_3, mask_1, mask_2, mask_3):
        # Feeding input to the model
        hidden_states_1, _ = self.layer(tensor_1, attention_mask = mask_1)
        hidden_states_2, _ = self.layer(tensor_2, attention_mask = mask_2)
        hidden_states_3, _ = self.layer(tensor_3, attention_mask = mask_3)
        
        # Get [CLS] token
        sentence_embedding_1 = hidden_states_1[:,0]
        sentence_embedding_2 = hidden_states_2[:,0]
        sentence_embedding_3 = hidden_states_3[:,0]

        # Calculate logit of sentence
        logit_1 = self.classifier(self.dropout(sentence_embedding_1))
        logit_2 = self.classifier(self.dropout(sentence_embedding_2))
        logit_3 = self.classifier(self.dropout(sentence_embedding_3))
        
        # Concatenate logits
        logits = torch.cat((logit_1, logit_2, logit_3), 1)
        
        return logits

class TaskB:
     
    def cal_accuracy(self, preds, labels):
        '''
        Takes prediction and true labels and return back accuracy
        '''
        pred = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()
        return np.sum(pred == labels) / len(labels)    

    def format_data_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train(self, model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, device, epochs=6):
        '''
        Train the model according to the inputs
        '''
        # ========================================
        #               Validation
        # ========================================
        loss_list = []

        for epoch_i in range(0, epochs):
            print("")
            print('********* Epoch {:} / {:} **********'.format(epoch_i + 1, epochs))
            print('Training...')

            t_start = time.time()

            total_loss = 0

            model.train()

            for step, batch in enumerate(train_dataloader):

                if step % 20 == 0 and not step == 0:
                    elapsed = self.format_data_time(time.time() - t_start)

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                input_id_1 = batch[0].to(device)
                input_id_2 = batch[1].to(device)
                input_id_3 = batch[2].to(device)
                input_mask_1 = batch[3].to(device)
                input_mask_2 = batch[4].to(device)
                input_mask_3 = batch[5].to(device)
                b_labels = batch[6].to(device)

                model.zero_grad()        

                outputs = model(input_id_1, input_id_2, input_id_3, input_mask_1, input_mask_2, input_mask_3)

                loss = criterion(outputs, torch.argmax(b_labels, dim=1))

                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)            

            loss_list.append(avg_train_loss)

            print("training loss: {0:.2f}".format(avg_train_loss))
            print("Training epoch time: {:}".format(self.format_data_time(time.time() - t_start)))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t_start = time.time()
            
            total_loss = 0

            model.eval()

            eval_loss, eval_accuracy = 0
            eval_accuracy = 0
            nb_eval_steps = 0
            nb_eval_examples = 0

            for batch in val_dataloader:

                batch = tuple(t.to(device) for t in batch)

                input_id_1, input_id_2, input_id_3, input_mask_1, input_mask_2, input_mask_3, b_labels, _ = batch
                
                with torch.no_grad():        
                    logits = model(input_id_1, input_id_2, input_id_3, input_mask_1, input_mask_2, input_mask_3)
                
                loss = criterion(logits, torch.argmax(b_labels, dim=1))

                total_loss += loss.item()
                
                logits = logits.detach().cpu().numpy()
                label_ids = torch.argmax(b_labels, dim=1)
                label_ids = label_ids.to('cpu').numpy()

                tmp_eval_accuracy = self.cal_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy

                nb_eval_steps += 1

            eval_loss = total_loss / len(val_dataloader)
            
            # Report the final accuracy for this validation run.
            print("Accuracy: {0:.3f}".format(eval_accuracy/nb_eval_steps))
            print("Average validation loss: {0:.4f}".format(eval_loss))
            print("Validation took: {:}".format(self.format_data_time(time.time() - t_start)))

        chkpt_dict = {'model_state_dist':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict()}
        
        torch.save(chkpt_dict, '../weights/'+'stB-roberta-ep-'+str(epoch_i+1)+'-vacc-'+str((100*eval_accuracy/nb_eval_steps))+'.pt')
        
        print("Training complete!")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='subtask-B')  
    parser.add_argument('--data-path', type=str, default='./data/')                              
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--eps', type=float, default=1e-8)
    args = parser.parse_args()

    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('Using the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    trainset = DatasetB(root=os.path.join(args.data_path, 'Training_Data'))
    valset = DatasetB(root=os.path.join(args.data_path, 'Dev_Data'))
    
    # Training logs
    os.makedirs('drive/My Drive/sem_eval/logs/', exist_ok=True)
    # Weight checkpoint
    os.makedirs('drive/My Drive/sem_eval/weights/', exist_ok=True)
    
    #Creating intsances of training and validation dataloaders
    train_dataloader = DataLoader(trainset, batch_size = args.batch_size, num_workers = 5, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size = 128, num_workers = 5, shuffle=False)
    
    model =  RobertaClassifier()
    
    criterion = nn.CrossEntropyLoss()
    
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # args.learning_rate - default is 5e-5
    # args.adam_epsilon  - default is 1e-8
    optimizer = AdamW(model.parameters(), lr = args.learning_rate, eps = args.eps)
    
    epochs = args.epochs
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(0.1*total_steps)
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)
    
    model.to(device)
    task_B = TaskB()                           
    task_B.train(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, device, epochs)
