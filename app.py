
# curl -d '{"thread_id": 1,"D": 5,"Q": 10000,"S": 200000}' https://11zwbpoixg.execute-api.us-east-1.amazonaws.com/default/my_incircle_and_shot_values 

# curl -d '{"question":"How to write a essay"' http://127.0.0.1:5000/articles

# curl -i -H "Content-Type: application/json" -X POST -d '{"question":"How to write a essay"' http://127.0.0.1:5000/articles

from flask import Flask, request, jsonify
import ast
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


app = Flask(__name__)
cred = credentials.Certificate("thekeystosuccess.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

@app.route("/ask",methods = ["POST","GET"])
def resources():
    if request.method == "POST":
        q = request.get_data().decode('utf-8')
        print("POST request", q )
        question = ast.literal_eval(q)["question"]

        encoded_input = tokenizer.encode(question, return_tensors='pt')
        print(encoded_input)
        # print(tokenizer.decode(encoded_input[0][2]))
        output = model.generate(encoded_input, max_length=1000,num_beams=5,no_repeat_ngram_size=2,early_stopping=True)
        text = tokenizer.decode(output[0], skip_special_tokens=True)

        db.collection("QsNAnswers").add(
            {"question":question,
            "answer":text
            }
        )

        return jsonify({
            "text": text
        })


# A welcome message to test our server
@app.route('/askedquestions',methods = ["GET"])
def askedquestions():
    stuff=[]
    docs = db.collection("QsNAnswers").get()
    # print(docs)
    for doc in docs:
        print(doc.to_dict())
        stuff.append(doc.to_dict())
        


    return jsonify({
            "text": stuff
        })


# A welcome message to test our server
@app.route('/')
def index():

    return "<h1> What's Poppn </h1>"




if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True,threaded=True, port=5000)
