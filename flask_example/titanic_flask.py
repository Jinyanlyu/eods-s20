from flask import Flask, escape, request, jsonify
import pickle as pkl
import pandas as pd

# need to train and pickle classifier
with open('./titanic_clf.pkl','rb') as f:
    clf = pkl.load(f)

app = Flask(__name__)

@app.route('/',methods=['POST'])
def predict():
    '''Try running:
r = requests.post('http://127.0.0.1:5000/',data={"pclass":1,"name":"Allen, Miss. Elisabeth Walton","sex":"female","age":29.0,"sibsp":0,"parch":0,"ticket":"24160","fare":211.3375,"cabin":"B5","embarked":"S","boat":"2","body":'',"home.dest":"St Louis, MO"})
r.text
'''
    prediction = None
    req_data = pd.DataFrame(request.form,index=[0])
    print(req_data,flush=True)
    if req_data is not None:
        prediction = clf.predict(req_data)
    if prediction:
        return jsonify([str(x) for x in prediction])
    else:
        return 'no predictions made'

if __name__ == '__main__':
    app.run()
