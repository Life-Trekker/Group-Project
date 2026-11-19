from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN, load_glove_matrix
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

#---- Recurrent Neural Network script (with GloVe) ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    RNN_TYPE = 'GRU'          # 'RNN', 'GRU', 'LSTM', or 'CUSTOM'
    USE_GLOVE = True
    GLOVE_DIM = 100
    #------------------------------------------------------

    #---- directory location section ----------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_4_data') + os.sep
    IMDB_DIR = os.path.join(DATA_DIR, 'text_classification') + os.sep
    DATA_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result', 'result_data') + os.sep
    GRAPH_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result', 'graphs') + os.sep
    GLOVE_PATH = os.path.join(DATA_DIR, 'glove', 'glove_100d.txt')
    os.makedirs(DATA_RESULT_DIR, exist_ok=True)
    os.makedirs(GRAPH_RESULT_DIR, exist_ok=True)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    loader = Dataset_Loader('imdb', '')
    loader.dataset_source_folder_path = IMDB_DIR
    loader.max_len = 400
    loader.max_vocab_size = 20000


    data = loader.load()
    vocab_size = len(loader.vocab)
    loader.load = lambda: data

    if USE_GLOVE:
        emb = load_glove_matrix(GLOVE_PATH, loader.vocab, GLOVE_DIM)
        rnn_method_obj = Method_RNN('RNN', '', vocab_size=vocab_size, embed_dim=GLOVE_DIM,
                                    rnn_type=RNN_TYPE,
                                    pretrained_embeddings=emb, freeze_embeddings=False)
        tag = f'{RNN_TYPE}_GLOVE'
    else:
        rnn_method_obj = Method_RNN('RNN', '', vocab_size=vocab_size, rnn_type=RNN_TYPE)
        tag = RNN_TYPE
    rnn_method_obj.max_epoch = 20

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = DATA_RESULT_DIR + tag + '_'
    result_obj.result_destination_file_name = 'imdb_prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------


    #---- running section ----------------------------------
    print('************ Start ************')

    setting_obj.prepare(loader, rnn_method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()

    print('************ Plot Convergence ************')
    h = rnn_method_obj.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(h['epoch'], h['loss'], marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax1.set_title('Training Loss Convergence')
    ax1.grid(True)

    ax2.plot(h['epoch'], h['train_acc'], marker='o', label='Train')
    ax2.plot(h['epoch'], h['test_acc'], marker='s', label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy per Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(GRAPH_RESULT_DIR + f'{tag.lower()}_convergence_curves.png', dpi=150)
    plt.show()


    print('************ Test Performance ************')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')

    print('************ Finish ************')
    # ------------------------------------------------------