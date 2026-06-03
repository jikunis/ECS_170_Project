from local_code.stage_4_code.Dataset_Loader_Classification import Dataset_Loader_Classification
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_1_code.Result_Saver import Result_Saver
import matplotlib.pyplot as plt

# ---- paths (update if your layout differs) ----
DATA_PATH   = '/Users/jonahkunis/Desktop/ECS_170_Project/data/stage_4_data/text_classification/'
RESULT_PATH = '/Users/jonahkunis/Desktop/ECS_170_Project/result/stage_4_result/'

# ---- 1. load data ----
data_loader = Dataset_Loader_Classification('IMDB', '')
data_loader.dataset_source_folder_path = DATA_PATH
data_loader.vocab_size  = 10000
data_loader.max_seq_len = 200
loaded_data = data_loader.load()

# ---- 2. build and run model ----
rnn = Method_RNN_Classification('RNN_Classification', '')
rnn.data          = loaded_data
rnn.rnn_type      = 'LSTM'   # try 'RNN', 'GRU' for 4-5
rnn.max_epoch     = 10
rnn.learning_rate = 1e-3
rnn.batch_size    = 64
rnn.embed_dim     = 128
rnn.hidden_dim    = 256
rnn.num_layers    = 2
rnn.dropout       = 0.5

result = rnn.run()

# ---- 3. evaluate ----
evaluator = Evaluate_Accuracy('evaluator', '')
evaluator.data = {'true_y': result['true_y'], 'pred_y': result['pred_y']}
metrics = evaluator.evaluate()
print('Test Results:', metrics)

# ---- 4. save metrics ----
saver = Result_Saver('saver', '')
saver.data = metrics
saver.result_destination_folder_path = RESULT_PATH
saver.result_destination_file_name   = 'IMDB_classification'
saver.fold_count = 0
saver.save()

# ---- 5. learning curve ----
plt.figure()
plt.plot(rnn.loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('IMDB LSTM Classification – Learning Curve')
plt.tight_layout()
plt.savefig(RESULT_PATH + 'IMDB_classification_learning_curve.png', dpi=150)
plt.close()
print('Learning curve saved.')
