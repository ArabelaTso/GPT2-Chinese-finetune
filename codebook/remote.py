from flask import Flask, render_template, request

from combine import setup_models, get_replies_from_two_models

pre_pt_filename = './models/prefix.json'
cont_pt_filename = './models/contain.json'
    
prefix_model, contain_model = setup_models(pre_pt_filename, cont_pt_filename)


def get_reply(prompt):
    results = get_replies_from_two_models(prefix_model, contain_model, prompt=prompt)
    print(results)
    if len(results) == 0:
        results = ['Not found.']
    return results

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_input():
    input_text = request.form['input']
    # return a string
    output_text = get_reply(input_text)
    
    return render_template('index3.html', input=input_text, output_list=output_text)

if __name__ == '__main__':
    app.run(host='localhost', port=8091)
    
    # app.run(host='localhost', port=9999)
    # app.run(host='10.79.219.111', port=5000)
    # app.run()
    