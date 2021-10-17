from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    request_method_str = request.method
    if request_method_str == "GET":
        return render_template('index.html', pic_name='static/base_pic0.svg')
    else:
        text = request.form["text"]
        random_str = uuid.uuid4().hex
        path = "app/static/" + random_str + ".svg"
        model = load("app/model.joblib")
        input_data = floats_string_to_input_arr(text)
        make_picture("app/age_data.pkl", model, input_data, path)

        return render_template('index.html', pic_name=path[4:])

# Till 24:00


def floats_string_to_input_arr(floats_str):
    floats = [float(x.strip()) for x in floats_str.split(',')]
    as_np_arr = np.array(floats).reshape(len(floats), 1)
    return as_np_arr


def make_picture(training_data_filename, model, new_inp_np_arr, output_file='predictions_pic.svg'):
    # Plot training data with model
    data = pd.read_pickle(training_data_filename)
    ages = data['Age']
    data = data[ages > 0]
    ages = data['Age']
    heights = data['Height']
    x_new = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (years)',
                                                                                 'y': 'Height (inches)'})

    fig.add_trace(go.Scatter(x=x_new.reshape(
        19), y=preds, mode='lines', name='Model'))

    new_preds = model.predict(new_inp_np_arr)

    fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs',
                  mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))

    fig.write_image(output_file, width=800, engine='kaleido')
    fig.show()
    # return fig


# if __name__ == "__main__":
#     app.run(debug=True)
