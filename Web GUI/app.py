from flask import Flask, render_template, request
import requests
import json

app = Flask(__name__)
paragraph = ""
question = ""


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', testing_display="block", result_display="none", paragraph=paragraph,
                           question=question)


def send_to_model(context, question, model):
    data = {
        "question": question,
        "context": context,
        "model": model,
    }

    data = json.dumps(data)
    headers = {"content-type": "application/json"}
    json_response = requests.post("http://localhost:8090/predict", data=data, headers=headers)
    response = json.loads(json_response.text)
    return response


@app.route('/', methods=["POST"])
def predict_form():
    global paragraph, question
    print(request.form)
    if request.form["Submit"] == "Predict":
        paragraph = request.form.get("paragraph")
        question = request.form.get("question")
        model = request.form.get("model")
        print("P: {}, Q: {}, M: {}".format(paragraph, question, model))
        response = send_to_model(paragraph, question, model)
        context = response["context"]

        if len(response["predicted_answer"]) > 0:
            # answer_start = response["context"].find(response["predicted_answer"])
            # answer_end = len(response["predicted_answer"]) + answer_start
            answer_start = response["answer_start"]
            answer_end = response["answer_end"]
            context = context[:answer_start] + "<span>" + context[answer_start:answer_end] + "</span>" + \
                      context[answer_end:]

        return render_template('home.html', result_context=context,
                               result_question=response["question"],
                               result_predicted_answer=response["predicted_answer"],
                               testing_display="none", result_display="block", paragraph=paragraph, question=question)
    elif request.form["Submit"] == "    Paragraph":
        paragraph = ""
        question = ""
        return render_template('home.html', testing_display="block", result_display="none", paragraph=paragraph,
                               question=question)
    elif request.form["Submit"] == "    Question":
        question = ""
        paragraph = request.form.get("paragraph")
        return render_template('home.html', testing_display="block", result_display="none", paragraph=paragraph,
                               question=question)

if __name__ == "__main__":
    app.run(debug=True)
