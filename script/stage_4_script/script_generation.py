from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_1_code.Result_Saver import Result_Saver
import matplotlib.pyplot as plt

# ---- paths (update if your layout differs) ----
# Expects a single .txt file, e.g. a story / novel corpus
DATA_FOLDER = '/Users/jonahkunis/Desktop/ECS_170_Project/data/stage_4_data/text_generation/'
DATA_FILE   = 'data'          # filename without extension (change to match yours)
RESULT_PATH = '/Users/jonahkunis/Desktop/ECS_170_Project/result/stage_4_result/'

# ---- 1. load data ----
data_loader = Dataset_Loader_Generation('TextGen', '')
data_loader.dataset_source_folder_path = DATA_FOLDER
data_loader.dataset_source_file_name   = DATA_FILE
data_loader.seq_len    = 30
data_loader.vocab_size = 5000
loaded_data = data_loader.load()

# ---- 2. build and run model ----
rnn = Method_RNN_Generation('RNN_Generation', '')
rnn.data          = loaded_data
rnn.rnn_type      = 'LSTM'   # try 'RNN', 'GRU' for 4-5
rnn.max_epoch     = 20
rnn.learning_rate = 1e-3
rnn.batch_size    = 128
rnn.embed_dim     = 128
rnn.hidden_dim    = 256
rnn.num_layers    = 2
rnn.dropout       = 0.3
rnn.gen_length    = 100
rnn.temperature   = 0.8      # lower = less random

metrics = rnn.run()
print('Test Metrics:', metrics)

# ---- 3. generate story ----
# change seed words to whatever makes sense for your corpus
SEED_WORDS = ['the', 'old', 'man']
generated_text = rnn.generate(seed_words=SEED_WORDS, gen_length=100)

print('\n--- Generated Story ---')
print(generated_text)
print('-----------------------\n')

# ---- 4. save generated text ----
gen_file = RESULT_PATH + 'generated_story.txt'
with open(gen_file, 'w') as f:
    f.write(f'Seed: {" ".join(SEED_WORDS)}\n\n')
    f.write(generated_text)
print(f'Generated story saved to {gen_file}')

# ---- 5. save perplexity / metrics ----
saver = Result_Saver('saver', '')
saver.data = metrics
saver.result_destination_folder_path = RESULT_PATH
saver.result_destination_file_name   = 'text_generation'
saver.fold_count = 0
saver.save()

# ---- 6. learning curve ----
plt.figure()
plt.plot(rnn.loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Text Generation LSTM – Learning Curve')
plt.tight_layout()
plt.savefig(RESULT_PATH + 'generation_learning_curve.png', dpi=150)
plt.close()
print('Learning curve saved.')
