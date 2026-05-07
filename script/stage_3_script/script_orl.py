from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_1_code.Result_Saver import Result_Saver
import matplotlib.pyplot as plt

# load data
data_loader = Dataset_Loader('ORL', '')
data_loader.dataset_source_folder_path = '/Users/arielpadovitz/Desktop/ECS170_Spring_2026_Source_Code_Template/data/stage_3_data/'
data_loader.dataset_source_file_name = 'ORL'
loaded_data = data_loader.load()

# set up and run model
cnn = Method_CNN_ORL('CNN_ORL', '')
cnn.data = loaded_data
result = cnn.run()

# evaluate
evaluator = Evaluate_Accuracy('evaluator', '')
evaluator.data = {'true_y': result['true_y'], 'pred_y': result['pred_y']}
metrics = evaluator.evaluate()
print('Test Results:', metrics)

# save results
saver = Result_Saver('saver', '')
saver.data = metrics
saver.result_destination_folder_path = '/Users/arielpadovitz/Desktop/ECS170_Spring_2026_Source_Code_Template/result/stage_3_result/'
saver.result_destination_file_name = 'ORL'
saver.fold_count = 0
saver.save()

# save learning curve
plt.plot(cnn.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ORL CNN Learning Curve')
plt.savefig('/Users/arielpadovitz/Desktop/ECS170_Spring_2026_Source_Code_Template/result/stage_3_result/ORL_learning_curve.png')
plt.close()