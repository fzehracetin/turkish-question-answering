import numpy as np
from tensorflow import keras
import tokenizers
from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import json

app = Flask(__name__)
api = Api(app)
CORS(app)


class Model:
    def __init__(self, max_len, path, model_name, tokenizer):
        self.max_len = max_len
        self.path = path
        self.model_path = self.path + model_name
        self.tokenizer = tokenizer
        self.model = keras.models.load_model(self.model_path)
        self.model_name = model_name

    def describe_class(self):
        print("Sequence length: {}\nPath: {} \nTokenizer: {}\nModel: {}".format(self.max_len,
                                                                                self.path,
                                                                                self.tokenizer,
                                                                                self.model_name))


class WikiElement:
    def __init__(self, question, context):
        self.question = question
        self.context = context
        self.input_ids = None
        self.token_type_ids = None
        self.attention_mask = None
        self.context_token_to_char = None

    def preprocess(self, model_class):
        # tokenize context
        tokenized_context = model_class.tokenizer.encode(self.context)
        # tokenize question
        tokenized_question = model_class.tokenizer.encode(self.question)
        # create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)
        # padding for equal length sequence
        padding_length = model_class.max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)  # len(input) [1] + padding [0]
            token_type_ids = token_type_ids + ([0] * padding_length)  # context [0] + question [1] + padding [0]
        elif padding_length < 0:
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets


def create_input_targets(element):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
    }
    for key in dataset_dict:
        dataset_dict[key].append(getattr(element, key))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
        ]
    return x


def predict_answer(question, context, model_class):
    # create wiki element object
    element = WikiElement(question, context)
    element.preprocess(model_class)
    # create input matrix for model
    x = create_input_targets(element)
    # predict
    predicted_start, predicted_end = model_class.model.predict(x)
    start = np.argmax(predicted_start)
    end = np.argmax(predicted_end)
    offsets = element.context_token_to_char
    predicted_char_start = offsets[start][0]

    if end < len(offsets):
        predicted_char_end = offsets[end][1]
        predicted_answer = element.context[predicted_char_start:predicted_char_end]
    else:
        predicted_answer = element.context[predicted_char_start:]

    result = {"question": element.question,
              "predicted_answer": predicted_answer,
              "context": element.context}

    return result


@app.route('/predict', methods=['POST'])
def handle_post_requests():
    data = request.get_json()
    print(data)
    model_class = None
    if data['model'] == "bert":
        model_class = bert_model_class
    # elif data['model'] == "electra":
    #     model_class = electra_model_class
    # elif data['model' = "albert"]:
    #     model_class = albert_model_class
    response = predict_answer(data['question'], data['context'], model_class)
    print(response["predicted_answer"])
    return json.dumps(response, ensure_ascii=False)


def init_bert():
    bert_max_len = 512
    bert_path = "bert_base_turkish_cased/"
    bert_model_name = "dbmdz-bert-base-turkish-cased_seqlen512_epochs10/"
    bert_tokenizer = tokenizers.BertWordPieceTokenizer(bert_path + "/vocab.txt", lowercase=False)

    bert_model_class = Model(bert_max_len, bert_path, bert_model_name, bert_tokenizer)
    return bert_model_class


def init_electra():
    electra_max_len = 512
    electra_path = "electra_base_turkish_cased_discriminator/"
    electra_model_name = ""
    electra_tokenizer = None


def init_albert():
    albert_max_len = 512
    albert_path = "albert_base_turkish_uncased"
    albert_model_name = ""
    albert_tokenizer = None


if __name__ == "__main__":
    bert_model_class = init_bert()
    # electra_model_class = init_electra()
    # albert_model_class = init_albert()
    app.run(debug=True, port=8090)
