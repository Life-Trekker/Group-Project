from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

#---- CNN script: ORL face dataset ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    #---- directory location section ----------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data') + os.sep
    DATA_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result', 'result_data') + os.sep
    GRAPH_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result', 'graphs') + os.sep
    os.makedirs(DATA_RESULT_DIR, exist_ok=True)
    os.makedirs(GRAPH_RESULT_DIR, exist_ok=True)
    #------------------------------------------------------

    # ---- object initialization section -------------------
    loader = Dataset_Loader('ORL', 'ORL face images, 40 subjects')
    loader.dataset_source_folder_path = DATA_DIR
    loader.dataset_source_file_name = 'ORL'
    loader.dataset_name = 'ORL'

    method_obj = Method_CNN_ORL('cnn orl', '', in_channels=1, image_size=(112, 92), num_classes=40)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = DATA_RESULT_DIR + 'CNN_'
    result_obj.result_destination_file_name = 'orl_prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------

    #---- running section ----------------------------------
    print('************ Start ************')

    setting_obj.prepare(loader, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()

    print('************ Plot Learning Curves ************')
    h = method_obj.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(h['epoch'], h['loss'], marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax1.set_title('ORL CNN - Training Loss')
    ax1.grid(True)

    ax2.plot(h['epoch'], h['train_acc'], marker='o', label='Train')
    ax2.plot(h['epoch'], h['test_acc'], marker='s', label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('ORL CNN - Accuracy per Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(GRAPH_RESULT_DIR + 'orl_convergence_curves.png', dpi=150)
    plt.show()

    print('************ Test Performance ************')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')

    print('************ Finish ************')
    # ------------------------------------------------------
