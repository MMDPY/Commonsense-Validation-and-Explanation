import pandas as pd


def data_loader(data_path, ans_path):
    df_a_data = pd.read_csv(data_path, index_col=0)
    df_a_data.reset_index(drop=True, inplace=True)
    df_a_ans = pd.read_csv(ans_path, header=None, index_col=0)
    df_a_ans.reset_index(drop=True, inplace=True)
    df_a = pd.concat([df_a_data, df_a_ans], axis=1)
    df_a.head()
    df_a.rename({1: 'label', 'sent0': 'sentence1', 'sent1': 'sentence2'}, axis=1, inplace=True)
    return df_a


def classification_data(mode):
    if mode == 'train':
        df = data_loader('../../data/Training_Data/subtaskA_data_all.csv',
                         '../../data/Training_Data/subtaskA_answers_all.csv')
    elif mode == 'dev':
        df = data_loader('../../data/Dev_Data/subtaskA_dev_data.csv',
                         '../../data/Dev_Data/subtaskA_gold_answers.csv')
    elif mode == 'test':
        df = data_loader('../../data/Test_Data/subtaskA_test_data.csv',
                         '../../data/Test_Data/subtaskA_gold_answers.csv')
    data_array = []
    for index, row in df.iterrows():
        first_sentence = row['sentence1']
        second_sentence = row['sentence2']
        first_label = 1 - row['label']
        second_label = row['label']
        data_array.extend([[first_sentence, first_label], [second_sentence, second_label]])

    df = pd.DataFrame(data_array, columns=['sentence', 'label'])
    return df
