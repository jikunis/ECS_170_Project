from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Result_Saver import Result_Saver
import matplotlib.pyplot as plt

# load data
data_loader = Dataset_Loader('mnist', '')
data_loader.dataset_source_folder_path = '/Users/arielpadovitz/Desktop/ECS170_Spring_2026_Source_Code_Template/data/stage_2_data/'
loaded_data = data_loader.load()

# set up and run model
mlp = Method_MLP('mlp', '')
mlp.data = loaded_data
result = mlp.run()

# evaluate
evaluator = Evaluate_Accuracy('evaluator', '')
evaluator.data = {'true_y': result['true_y'], 'pred_y': result['pred_y']}
metrics = evaluator.evaluate()
print('Test Results:', metrics)

# save learning curve plot
plt.plot(mlp.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Convergence Curve')
plt.savefig('../../result/stage_2_result/learning_curve.png')
