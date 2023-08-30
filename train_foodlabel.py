import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_model import CNN

##### 1. Parameter #####

# WORD-level
MAX_NUM_WORDS  = 5131
EMBEDDING_DIM  = 100
MAX_SEQ_LENGTH = 350
KERNEL_SIZES   = [3,4,5]
FEATURE_MAPS   = [50,50,50]

# GENERAL
DROPOUT_RATE = 0.5
HIDDEN_UNITS = 100
NB_CLASSES   = 2

# LEARNING
BATCH_SIZE = 100
NB_EPOCHS  = 10
RUNS       = 5
VAL_SIZE   = 0.2

##### 2. Read input files #####

embedding_matrix = np.zeros((MAX_NUM_WORDS + 1, EMBEDDING_DIM))
n_line = 1
with open(f"data/word_emb", encoding='utf-8') as we:
    for line in we.readlines():
        values = line.split()
        embedding_matrix[n_line] = np.asarray(values)
        n_line = n_line + 1

em = tf.keras.layers.Embedding(
    input_dim    = MAX_NUM_WORDS + 1,
    output_dim   = EMBEDDING_DIM,
    input_length = MAX_SEQ_LENGTH,
    weights      = [embedding_matrix],
    trainable    = True,
    name         = "word_embedding"
)
tr_wakati = pd.read_csv(f"data/tr_wakati", sep="\t", header=None)
tr_label = pd.read_csv(f"data/tr_label")
tr_label = tr_label["label"].to_numpy() 

##### 3. Traning #####

random_state = np.random.randint(100)
_X_train, _X_val, _y_train, _y_val = train_test_split(
    tr_wakati, tf.keras.utils.to_categorical(tr_label),
    test_size    = VAL_SIZE, 
    random_state = random_state
)
emb_layer = em
model = CNN(
    embedding_layer    = emb_layer,
    num_words          = MAX_NUM_WORDS + 1,
    embedding_dim      = EMBEDDING_DIM,
    kernel_sizes       = KERNEL_SIZES,
    feature_maps       = FEATURE_MAPS,
    max_seq_length     = MAX_SEQ_LENGTH,
    dropout_rate       = DROPOUT_RATE,
    hidden_units       = HIDDEN_UNITS,
    nb_classes         = NB_CLASSES
).build_model()
model.compile(
    loss      = "categorical_crossentropy", #  "binary_crossentropy",
    optimizer = tf.optimizers.Adam(), 
    metrics   = ["accuracy"]
)
history = model.fit(
    _X_train, _y_train,
    epochs          = NB_EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_data = (_X_val, _y_val),
    callbacks       = [tf.keras.callbacks.ModelCheckpoint(
        filepath       = f"model.h5",
        monitor        = "val_loss",
        verbose        = 1,
        save_best_only = True,
        mode           = "min"
    )]
)

##### 4. Prediction #####

test_wakati = pd.read_csv(f"data/test_wakati", sep="\t", header=None)
test_label = pd.read_csv(f"data/test_label")
test_label = test_label["label"].to_numpy() 

cnn_ = tf.keras.models.load_model(f"model.h5")
score = cnn_.evaluate(test_wakati, tf.keras.utils.to_categorical(test_label), verbose=0)

print(f"Running test with model: {score[0]} loss / {score[1]} acc")

## p = cnn_.predict(adf_wakati)
## np.savetxt(f"tmp/pr_out.csv", p)
