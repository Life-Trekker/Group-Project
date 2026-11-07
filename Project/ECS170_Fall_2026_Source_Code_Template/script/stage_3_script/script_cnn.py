from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
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
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data') + os.sep
    DATA_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result', 'result_data') + os.sep
    GRAPH_RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result', 'graphs') + os.sep
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    loader = Dataset_Loader('loader', '')
    loader.dataset_source_folder_path = DATA_DIR
    
    # load MNIST dataset
    loader.dataset_source_file_name = 'CIFAR'
    loader.dataset_name = 'CIFAR'



    mnist_method_obj = Method_CNN_MNIST('cnn mnist', '')
    cifar_method_obj = Method_CNN_CIFAR('cnn cifar', '')
    orl_method_obj = Method_CNN_ORL('cnn orl', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = DATA_RESULT_DIR + 'CNN_'
    result_obj.result_destination_file_name = 'cifar_prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------


    #---- running section ----------------------------------
    print('************ Start ************')


    setting_obj.prepare(loader, cifar_method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()

    print('************ Plot Convergence ************')
    h = cifar_method_obj.history

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
    plt.savefig(GRAPH_RESULT_DIR + 'cifar_convergence_curves.png', dpi=150)
    plt.show()
    
    
    print('************ Test Performance ************')
    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')
        
    print('************ Finish ************')
    # ------------------------------------------------------