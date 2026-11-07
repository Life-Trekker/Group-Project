from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    #---- directory location section ----------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_2_data') + os.sep
    DATA_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result', 'result_data') + os.sep
    GRAPH_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result', 'graphs') + os.sep
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('train', '')
    data_obj.dataset_source_folder_path = DATA_DIR
    data_obj.dataset_train_source_file_name = 'train.csv'
    data_obj.dataset_test_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = DATA_RESULT_DIR + 'MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------


    #---- running section ----------------------------------
    print('************ Start ************')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()

    print('************ Plot Convergence ************')
    h = method_obj.history

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
    plt.savefig(GRAPH_RESULT_DIR + 'convergence_curves.png', dpi=150)
    plt.show()
    
    
    print('************ Test Performance ************')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')
        
    print('************ Finish ************')
    # ------------------------------------------------------