from flask import Flask, render_template, request

from prefix_inferring import setup

trie = setup()

def get_reply(prompt):
    results = trie.search(prompt)
    # print(results)
    if len(results) == 0:
        result = 'Not found.'
    else:
        result = results[0][0]
    return result

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_input():
    input_text = request.form['input']
    output_text = get_reply(input_text) #'Hello ' + input_text
    return render_template('index.html', input=input_text, output=output_text)

if __name__ == '__main__':
    app.run(host='localhost', port=8090)
    
    # app.run(host='localhost', port=9999)
    # app.run(host='10.79.219.111', port=5000)
    # app.run()
    