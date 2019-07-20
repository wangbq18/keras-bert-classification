import os
import numpy as np
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.layers import Dense, Input,concatenate, Flatten
from keras.models import Model
from keras.engine.topology import Layer
from bert import tokenization

################# Class ########################

class BertClassification(object):


    def __init__(self, BERT_PRETRAINED_DIR):

        self.bert_model = get_pretrained_model(BERT_PRETRAINED_DIR, maxlen)

    def get_text_classification_model(self, LABEL_LEN, OUTPUT_ACTIVATION):

        layer9 = self.bert_model.get_layer('Encoder-9-FeedForward-Norm').output
        layer10 = self.bert_model.get_layer('Encoder-10-FeedForward-Norm').output
        layer11 = self.bert_model.get_layer('Encoder-11-FeedForward-Norm').output
        layer12 = self.bert_model.get_layer('Encoder-12-FeedForward-Norm').output

        concatenate_layer = concatenate([layer9, layer10, layer11, layer12])

        pool_output = Dense(LABEL_LEN, activation=OUTPUT_ACTIVATION,name = 'real_output')(concatenate_layer)

        model  = Model(inputs=self.bert_model.input, outputs=pool_output)

        return model

    def get_text_classification_model_with_extra_input(self,extra_input_len, LABEL_LEN, OUTPUT_ACTIVATION):

        layer9 = self.bert_model.get_layer('Encoder-9-FeedForward-Norm').output
        layer10 = self.bert_model.get_layer('Encoder-10-FeedForward-Norm').output
        layer11 = self.bert_model.get_layer('Encoder-11-FeedForward-Norm').output
        layer12 = self.bert_model.get_layer('Encoder-12-FeedForward-Norm').output

        concatenate_layer = concatenate([layer9, layer10, layer11, layer12])
        concatenate_layer = NonMaskingLayer()(concatenate_layer)
        flattened = Flatten()(concatenate_layer)

        l1_input = Input(shape=(extra_input_len,), name='l1_input')
        conc = concatenate([flattened, l1_input], axis=1)

        pool_output = Dense(LABEL_LEN, activation= OUTPUT_ACTIVATION,name = 'real_output')(conc)

        model  = Model(inputs=[self.bert_model.layers[0].input, self.bert_model.layers[1].input,l1_input], outputs=pool_output)

        return model


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


################## Functions ####################

def get_pretrained_model(BERT_PRETRAINED_DIR, maxlen):

    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True,seq_len=maxlen)

    return model

def get_text_classification_model(BERT_PRETRAINED_DIR, maxlen, LABEL_LEN, OUTPUT_ACTIVATION):
    bert_model = get_pretrained_model(BERT_PRETRAINED_DIR, maxlen)
    sequence_output  = model.layers[-6].output
    pool_output = Dense(LABEL_LEN, activation=OUTPUT_ACTIVATION,name = 'real_output')(sequence_output)

    model  = Model(inputs=bert_model.input, outputs=pool_output)

    return model

def get_text_classification_model_with_extra_input(BERT_PRETRAINED_DIR,extra_input_len, maxlen, LABEL_LEN, OUTPUT_ACTIVATION):

    bert_model = get_pretrained_model(BERT_PRETRAINED_DIR, maxlen)
    sequence_output  = bert_model.layers[-6].output
    l1_input = Input(shape=(extra_input_len,), name='l1_input')
    conc = concatenate([sequence_output, l1_input])
    pool_output = Dense(LABEL_LEN, activation= OUTPUT_ACTIVATION,name = 'real_output')(conc)

    model  = Model(inputs=[bert_model.layers[0].input, bert_model.layers[1].input,l1_input], outputs=pool_output)

    return model

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for i in range(example.shape[0]):
        tokens_a = tokenizer.tokenize(example[i])
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)



def perform_pre_processing(BERT_PRETRAINED_DIR, maxlen, text_lines):

    dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)

    token_input = convert_lines(text_lines,maxlen,tokenizer)
    seg_input = np.zeros((token_input.shape[0],maxlen))
    mask_input = np.ones((token_input.shape[0],maxlen))

    return token_input, seg_input, mask_input