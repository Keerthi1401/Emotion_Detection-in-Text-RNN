import datetime
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Credits: Armin Seyeditabari & Narges Tabari.
# Updated and refactored code by Vinit Shah.

PUNCTS = [",", ".", """, ":", ")", "(", "-", "!", "?", "|", ";", """, "$", "&", "/", "[", "]", ">",
          "%", "=", "#", "*", "+", "\\", "•",  "~", "@", "£", "·", "_", "{", "}", "©", "^", "®",
          "`",  "<", "→", "°", "€", "™", "›",  "♥", "←", "×", "§", "″", "′", "Â", "█", "½", "à",
          "…", "“", "★", "”", "–", "●", "â", "►", "−", "¢", "²", "¬", "░", "¶", "↑", "±", "¿", "▾",
          "═", "¦", "║", "―", "¥", "▓", "—", "‹", "─", "▒", "：", "¼", "⊕", "▼", "▪", "†", "■",
          "’", "▀", "¨", "▄", "♫", "☆", "é", "¯", "♦", "¤", "▲", "è", "¸", "¾", "Ã", "⋅", "‘", "∞",
          "∙", "）", "↓", "、", "│", "（", "»", "，", "♪", "╩", "╚", "³", "・", "╦", "╣", "╔", "╗",
          "▬", "❤", "ï", "Ø", "¹", "≤", "‡", "√"]


def get_sentences(data):
    """ Extracts sentences from file """
    with open(data, 'r') as f:
        sentences = f.read().splitlines()
    return sentences


def clean_text(data):
    """ Seperates punctuations from words in given string data """
    data = str(data).strip()
    for punct in PUNCTS:
        data = data.replace(punct, " %s " % punct)
    data = data.replace(",", " ")
    data = data.replace("\n", " ")
    data = data.lower()
    text = re.sub(r"( #\S+)*$", "", data)
    return text


def prepare_vocab(max_features, token_data):
    """ Prepares the original vocabulary """
    # token_data["text"] = token_data["text"].apply(lambda x: clean_text(x))
    tokens_text = token_data["text"].fillna("_##_").values
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(tokens_text))
    return tokenizer


def prepare_test(test_dataset, tokenizer):
    """ Prepares test data based on vocabulary of training data """
    # cleans up the text and makes it lower case
    all_X = [clean_text(data) for data in test_dataset]
    all_X = tokenizer.texts_to_sequences(all_X)
    all_X = tf.keras.preprocessing.sequence.pad_sequences(all_X, maxlen=maxlen)
    return all_X


def load_embedding(word_index, embedding_file, max_features):
    """
    Create an embedding matrix in which we keep only the embeddings 
    for words which are in our word_index
    """
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype="float32")
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf-8"))
    embed_size = len(embeddings_index[next(iter(embeddings_index))])
    # make sure all embeddings have the right format
    key_to_del = []
    for key, value in embeddings_index.items():
        if not len(value) == embed_size:
            key_to_del.append(key)
    for key in key_to_del:
        del embeddings_index[key]
    words_not_found = []
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    count = 0
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count = count + 1
        else:
            words_not_found.append(word)
    # with open("words_not_found.txt", "w+", encoding="utf-8") as f:
    #     for item in words_not_found:
    #         f.write("%s\n" % item)
    return embedding_matrix, embed_size


def dot_product(data, kernel):
    """ Wrapper for dot product operation"""
    return tf.keras.backend.squeeze(
        tf.keras.backend.dot(data, tf.keras.backend.expand_dims(kernel)), axis=-1)


class AttentionWithContext(tf.keras.layers.Layer):
    """ Reference https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2 """

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = tf.keras.initializers.get("glorot_uniform")

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.u_constraint = tf.keras.constraints.get(u_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "W_regularizer": self.W_regularizer,
                "u_regularizer": self.u_regularizer,
                "b_regularizer": self.b_regularizer,
                "W_constraint": self.W_constraint,
                "u_constraint": self.u_constraint,
                "b_constraint": self.b_constraint,
                "bias": self.bias,
            }
        )
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name="{}_W".format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer="zero",
                                     name="{}_b".format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name="{}_u".format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = tf.keras.activations.tanh(uit)
        ait = dot_product(uit, self.u)
        a = tf.keras.backend.exp(ait)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= tf.keras.backend.cast(mask, tf.keras.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN"s. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= tf.keras.backend.cast(tf.keras.backend.sum(a, axis=1, keepdims=True) +
                                   tf.keras.backend.epsilon(),
                                   tf.keras.backend.floatx())
        a = tf.keras.backend.expand_dims(a)
        weighted_input = x * a
        return tf.keras.backend.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def model_gru_att(embedding_matrix, embed_size, max_features):
    """ GRU With attention model """
    inp = tf.keras.layers.Input(shape=(maxlen,))
    x = tf.keras.layers.Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(35, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    conc = tf.keras.layers.Dense(35, activation="relu")(x)
    conc = tf.keras.layers.Dropout(0.5)(conc)
    outp = tf.keras.layers.Dense(1, activation="sigmoid")(conc)
    model = tf.keras.models.Model(inputs=inp, outputs=outp)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# MAIN
embedding_file = "vectors/crawl-300d-2M-2.txt"
training_token_dataset = "data/wang_cleaned_full_dataset.csv"
curr_dt = datetime.datetime.now().strftime("%m%d%y_%H%M%S")
maxlen = 35
embeddingSize = 300

joy_weight_file = "models/joy-ft-emem2.h5"
sadness_weight_file = "models/sadness-ft-emem2.h5"
anger_weight_file = "models/anger-ft-emem2.h5"
love_weight_file = "models/love-ft-emem2.h5"
thankfulness_weight_file = "models/thankfulness-ft-emem2.h5"
fear_weight_file = "models/fear-ft-emem2.h5"
# surprise_weight_file = "models/surprise-250-20.h5"

novel_path = os.path.abspath(
    (os.path.join(os.path.dirname(__file__), "novels_cleaned_sentences")))
output_path = os.path.abspath(
    (os.path.join(os.path.dirname(__file__), "novels_prediction")))

files = [os.path.join(novel_path, f) for f in os.listdir(novel_path)
         if os.path.isfile(os.path.join(novel_path, f))]

print("Loading Training data...")
token_data = pd.read_csv(training_token_dataset)

print("Preparing tokenizer data...")
tknzr_100k = prepare_vocab(100000, token_data)
tknzr_50k = prepare_vocab(50000, token_data)
# tknzr_25k = prepare_vocab(25000, token_data)

print("Preparing models....")
embedding_matrix_100k, embedding_size = load_embedding(tknzr_100k.word_index, embedding_file, 100000)
model_joy = model_gru_att(embedding_matrix_100k, embeddingSize, 100000)
model_sadness = model_gru_att(embedding_matrix_100k, embeddingSize, 100000)
model_anger = model_gru_att(embedding_matrix_100k, embeddingSize, 100000)
model_love = model_gru_att(embedding_matrix_100k, embeddingSize, 100000)

embedding_matrix_50k, embedding_size = load_embedding(tknzr_50k.word_index, embedding_file, 50000)
model_thankfulness = model_gru_att(embedding_matrix_50k, embeddingSize, 50000)
model_fear = model_gru_att(embedding_matrix_50k, embeddingSize, 50000)

# embedding_matrix_25k, embedding_size = load_embedding(tknzr_25k.word_index, embedding_file, 25000)
# model_surprise = model_gru_att(embedding_matrix_25k, embeddingSize, 25000)

print("Loading models...")
model_joy.load_weights(joy_weight_file)
model_sadness.load_weights(sadness_weight_file)
model_anger.load_weights(anger_weight_file)
model_love.load_weights(love_weight_file)
model_thankfulness.load_weights(thankfulness_weight_file)
model_fear.load_weights(fear_weight_file)
# model_surprise.load_weights(surprise_weight_file)

for f in files:
    test_token_dataset = f
    file_basename = os.path.basename(f)
    output_dataset = os.path.join(output_path, file_basename)

    print("Loading Test data...")
    test_dataset = pd.read_csv(test_token_dataset, delimiter="|", names=["text"])
    test_dataset = test_dataset["text"].tolist()
    print(f"Novel: {test_token_dataset}")

    print("Generating predictions...")
    test_X_100k = prepare_test(test_dataset, tknzr_100k)
    test_X_50k = prepare_test(test_dataset, tknzr_50k)
    # test_X_25k = prepare_test(test_dataset, tknzr_25k)

    print("Predicting Joy...")
    pred_joy_y = model_joy.predict([test_X_100k], batch_size=1024, verbose=0)
    joy_preds = pred_joy_y.tolist()
    joy_preds = [j for sub in joy_preds for j in sub]

    print("Predicting Sadness...")
    pred_sadness_y = model_sadness.predict([test_X_100k], batch_size=1024, verbose=0)
    sadness_preds = pred_sadness_y.tolist()
    sadness_preds = [j for sub in sadness_preds for j in sub]

    print("Predicting Anger...")
    pred_anger_y = model_anger.predict([test_X_100k], batch_size=1024, verbose=0)
    anger_preds = pred_anger_y.tolist()
    anger_preds = [j for sub in anger_preds for j in sub]

    print("Predicting Love...")
    pred_love_y = model_love.predict([test_X_50k], batch_size=1024, verbose=0)
    love_preds = pred_love_y.tolist()
    love_preds = [j for sub in love_preds for j in sub]

    print("Predicting Thankfulness...")
    pred_thankfulness_y = model_thankfulness.predict([test_X_50k], batch_size=1024, verbose=0)
    thankfulness_preds = pred_thankfulness_y.tolist()
    thankfulness_preds = [j for sub in thankfulness_preds for j in sub]

    print("Predicting Fear...")
    pred_fear_y = model_fear.predict([test_X_50k], batch_size=1024, verbose=0)
    fear_preds = pred_fear_y.tolist()
    fear_preds = [j for sub in fear_preds for j in sub]

    # print("Predicting Surprise...")
    # pred_surprise_y = model_surprise.predict([test_X_25k], batch_size=1024, verbose=0)
    # surprise_preds = pred_surprise_y.tolist()
    # surprise_preds = [j for sub in surprise_preds for j in sub]

    print("Exporting results...")
    resultsdict = {
        "text": test_dataset,
        "joy": joy_preds,
        "sadness": sadness_preds,
        "anger": anger_preds,
        "love": love_preds,
        "thankfulness": thankfulness_preds,
        "fear": fear_preds,
        # "surprise": surprise_preds
    }

    results_df = pd.DataFrame(resultsdict)
    results_df.to_csv(output_dataset, float_format="%.3f", index=False)
