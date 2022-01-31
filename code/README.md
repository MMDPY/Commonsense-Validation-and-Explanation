## Installation and Execution Instructions for Subtask A - Machine learning models
- Clone the repository
- Open a terminal inside the cloned folder (the folder should consist of code, data, reports, runs, requirements.txt and README.md)
- Run command ```python3 -m pip install -r requirements.txt ``` to install all the required packages.
- Run the command ```python3 code/ML-models/main.py``` to run logistic regression, naive bayes and SVM on subtask A.
- Run the command ```python3 ./code/pretrained/main.py ./data/Test_Data/subtaskA_test_data.csv ./data/Test_Data/subtaskA_gold_answers.csv ./runs/gpt.csv gpt``` to run 

## Installation and Execution Instructions for Subtask A - Pretrained language models
- Clone the repository
- Open a terminal inside the cloned folder (the folder should consist of code, data, reports, runs, requirements.txt and README.md)
- Run command ```python3 -m pip install -r requirements.txt ``` to install all the required packages.
- Change your current directory to from the root directory to code/pretrained by using cd command.

- To run gpt2 language model:

- Run the command ``` python 3 main.py gpt ``` 

- To run bert base language model:
- Run the command ``` python 3 main.py bert ``` 

- To fine-tune bert on the dataset and test on the test set:
- Run the command ``` python 3 main.py tuned_bert ``` 

- To load the weights from the checkpoint directory and run the best performing model:
- Run the command ``` python 3 main.py load ``` 

## Installation and Execution Instructions for Subtask B - Pretrained language models

Run command python3 code/subtaskB.py --data-path ./data/ --batch-size 32 --epochs 1 --learning-rate 2e-5 --eps 1e-8 to run RoBERTa language model for subtask B.
