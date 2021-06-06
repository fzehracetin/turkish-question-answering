import numpy as np
from tensorflow import keras
import tokenizers
from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import json
from transformers import ElectraTokenizerFast, AlbertTokenizerFast

app = Flask(__name__)
api = Api(app)
CORS(app)


class Model:
    def __init__(self, max_len, path, model_name, tokenizer, model_id):
        self.max_len = max_len
        self.path = path
        self.model_path = self.path + model_name
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = keras.models.load_model(self.model_path, compile=False)
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
        if model_class.model_id == "bert":
            tokenized_context = model_class.tokenizer.encode(self.context)
            tokenized_question = model_class.tokenizer.encode(self.question)
        else:
            tokenized_context = model_class.tokenizer(self.context, return_offsets_mapping=True)
            tokenized_question = model_class.tokenizer(self.question, return_special_tokens_mask=True)

            tokenized_context.offsets = tokenized_context.offset_mapping
            tokenized_context.ids = tokenized_context.input_ids
            tokenized_question.ids = tokenized_question.input_ids
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

    if start >= len(offsets):
        print("ERROR: Answer couldn't extracted!")
        result = {"question": element.question,
                  "predicted_answer": "",
                  "context": element.context}
        return result

    predicted_char_start = offsets[start][0]
    if end < len(offsets):
        predicted_char_end = offsets[end][1]
        predicted_answer = element.context[predicted_char_start:predicted_char_end]
    else:
        predicted_char_end = len(element.context)
        predicted_answer = element.context[predicted_char_start:]

    if len(predicted_answer) > 0 and predicted_answer[0] == ' ':
        predicted_answer = predicted_answer[1:]

    result = {"question": element.question,
              "predicted_answer": predicted_answer,
              "context": element.context,
              "answer_start": predicted_char_start,
              "answer_end": predicted_char_end
              }

    return result


@app.route('/predict', methods=['POST'])
def handle_post_requests():
    data = request.get_json()
    print(data)
    model_class = None
    if data['model'] == "Bert":
        model_class = bert_model_class
    elif data['model'] == "Electra":
        model_class = electra_model_class
    elif data['model'] == "Albert":
        model_class = albert_model_class
    response = predict_answer(data['question'], data['context'], model_class)
    print(response["predicted_answer"])
    return json.dumps(response, ensure_ascii=False)


def init_bert():
    bert_max_len = 512
    bert_path = "bert_base_turkish_cased/"
    bert_model_name = "dbmdz-bert-base-turkish-cased_seqlen512_epochs10/"
    bert_tokenizer = tokenizers.BertWordPieceTokenizer(bert_path + "/vocab.txt", lowercase=False)
    bert_model_class = Model(bert_max_len, bert_path, bert_model_name, bert_tokenizer, "bert")
    print("1. BERT LOADED")
    return bert_model_class


def init_electra():
    electra_max_len = 512
    electra_path = "electra_base_turkish_cased_discriminator/"
    electra_model_name = "dbmdz-electra-base-turkish-cased-discriminator_seqlen512_bacth64_epochs15/"
    electra_tokenizer = ElectraTokenizerFast.from_pretrained(electra_path, do_lower_case=False)
    electra_model_class = Model(electra_max_len, electra_path, electra_model_name, electra_tokenizer, "electra")
    print("2. ELECTRA LOADED")
    return electra_model_class


def init_albert():
    albert_max_len = 512
    albert_path = "albert_base_turkish_uncased/"
    albert_model_name = "loodos-albert-base-turkish-uncased_seqlen512_batch64_epochs10/"
    albert_tokenizer = AlbertTokenizerFast.from_pretrained(albert_path, do_lower_case=False, keep_accents=True)
    albert_model_class = Model(albert_max_len, albert_path, albert_model_name, albert_tokenizer, "albert")
    print("3. ALBERT LOADED")
    return albert_model_class


if __name__ == "__main__":
    bert_model_class = init_bert()
    electra_model_class = init_electra()
    albert_model_class = init_albert()
    print("4. READY TO SERVE")
    app.run(debug=False, use_reloader=False, port=8090)
