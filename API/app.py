import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelPickle.pkl','rb'))
scaler = pickle.load(open('fitData.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    val=request.form.values()
    # we need to transform the answers into the format needed to apply the model
    val2=[x for x in val]
    data2=[]
    # separate the values of the 30 derivated that were in 1 case
    for ele in val2:
        if ',' in ele:
            un=ele.split(',')
            for y in un:
                data2.append(float(y))
        else:
            temp_ele=float(ele)
        data2.append(temp_ele)
    day=int(data2[0])
    data2.pop(0)
    # make the corresponding day as 1 in a 7 floats matrix
    corresp_days=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    corresp_days[day]=1.0
    data2[:0]=corresp_days
    # calculate the meanCC 
    meanCC=((data2[41]+data2[40]+data2[39]) / 3)
    data2.insert(43,meanCC)
    cat_current=int(data2[2]) - 1
    categ_empty=[0 for i in range(106)]
    categ_empty[cat_current]=1
    data2.pop()
    data2.extend(categ_empty)
    # scale data
    final_features = [np.array(data2)]
    final_data=scaler.transform(final_features)
    # make the prediction and return
    prediction = model.predict(final_data)
    output = round(prediction[0], 2)
    return render_template('prediction.html', prediction_text='Number of comments estimated: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)