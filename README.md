[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=5740372&assignment_repo_type=AssignmentRepo)
## Participants:
_Ali Zamani_ [azamani1@ualberta.ca] 

_Mohammad Karimiabdolmaleki_ [karimiab@ualberta.ca]

# Commonsense Validation and Explanation (ComVE)

SemEval 2020 Task 4: Commonsense Validation and Explanation.

## Introduction

"The task is to directly test whether a system can differentiate natural language statements that make sense from those that do not make sense. We designed three subtasks. The first task is to choose from two natural language statements with similar wordings which one makes sense and which one does not make sense; The second task is to find the key reason from three options why a given statement does not make sense;."

The detailed description of the task can be found in [Task Proposal](reports/karimiab_azamani1_proposal.pdf).

### Example

#### Task A: Commonsense Validation

Which statement of the two is against common sense?

- Statement 1: He put a turkey into the fridge. *(correct)*
- Statement 2: He put an elephant into the fridge.

#### Task B: Commonsense Explanation (Multi-Choice)

Select the most corresponding reason why this statement is against common sense.

- Statement: He put an elephant into the fridge.

- Reasons:

  - **A**: An elephant is much bigger than a fridge. *(correct)*
  - **B**: Elephants are usually white while fridges are usually white.
  - **C**: An elephant cannot eat a fridge.

## Evaluation

Subtask A and B will be evaluated using **accuracy**. 

## Citation

```bib
@inproceedings{wang-etal-2020-semeval,
    title = "{S}em{E}val-2020 Task 4: Commonsense Validation and Explanation",
    author = "Wang, Cunxiang  and
      Liang, Shuailong  and
      Jin, Yili  and
      Wang, Yilong  and
      Zhu, Xiaodan  and
      Zhang, Yue",
    booktitle = "Proceedings of The 14th International Workshop on Semantic Evaluation",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

```bib
 @inproceedings{wang-etal-2019-make,
    title = "Does it Make Sense? And Why? A Pilot Study for Sense Making and Explanation",
    author = "Wang, Cunxiang  and
      Liang, Shuailong  and
      Zhang, Yue  and
      Li, Xiaonan  and
      Gao, Tian",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1393",
    pages = "4020--4026",
    abstract = "Introducing common sense to natural language understanding systems has received increasing research attention. It remains a fundamental question on how to evaluate whether a system has the sense-making capability. Existing benchmarks measure common sense knowledge indirectly or without reasoning. In this paper, we release a benchmark to directly test whether a system can differentiate natural language statements that make sense from those that do not make sense. In addition, a system is asked to identify the most crucial reason why a statement does not make sense. We evaluate models trained over large-scale language modeling tasks as well as human performance, showing that there are different challenges for system sense-making.",
}
 ```

## Installation and Execution Instructions
- Clone the repository
- Open a terminal inside the cloned folder (the folder should consist of code, data, reports, runs, requirements.txt and README.md)
- Run command ```python3 -m pip install -r requirements.txt ``` to install all the required packages.
- Run command ```python3 code/subtaskB.py --data-path ./data/ --batch-size 32 --epochs 1 --learning-rate 2e-5 --eps 1e-8 ``` to run RoBERTa language model for subtask B.
- Run the command ```python3 code/ML-models/main.py``` to run logistic regression, naive bayes and SVM on subtask A.
- Run command ```cd code/pretrained``` to go the pretrained directory
- Run command ```python 3 main.py gpt``` to run gpt2 model for subtask A.
- Run command ```python 3 main.py bert``` to run bert language model for subtask A.
- Run command ```python 3 main.py tuned_bert``` to fine-tune bert on the dataset and test on the test set for subtask A.
- Run command ```python 3 main.py load``` to load the weights from the checkpoint directory and run the best performing model (download files from https://drive.google.com/drive/folders/1NydG2Uiz7_hGJYJ0mp3nupqxKA2Q1TSz?usp=sharing and save them in the checkpoint directory) for subtask A
- Note that if you are using a Windows machine, you need to replace all of the slashes (/) with backslash (\\).
- The results and figures can be found inside [runs](runs)


## Resources
https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=nskPzUM084zL https://medium.com/@aksun/i-put-an-elephant-in-my-fridge-using-nlp-techniques-to-recognize-the-absurdity-2d8d565659e https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128 https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
https://github.com/soumyardash/SemEval2020-Task4.git
