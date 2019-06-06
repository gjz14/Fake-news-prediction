from flask import Flask, render_template, request
from datetime import timedelta


app = Flask(__name__)

app.config['DEBUG'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.config['TEMPLATES_AUTO_RELOAD'] = True



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model_explanation/', methods=['GET', 'POST'])
def model_explanation():
    if request.method == 'GET':
        return render_template('model_explanation.html')
    else:
        pass


@app.route('/part2/', methods=['GET', 'POST'])
def part2():
    if request.method == 'GET':
        return render_template('part2.html')
    else:
        pass

@app.route('/part1/', methods=['GET', 'POST'])
def part1():
    if request.method == 'GET':
        return render_template('part1.html')
    else:
        return render_template('part1_res.html')


@app.route('/part1_res/', methods=['GET', 'POST'])
def part1_res():
    if request.method == 'GET':
        pass
    else:
        res = request.form['name']
        num = int(request.form['num'])
        from interface_model1 import app
        import time
        args = app(res, num)
        args['input'] = res
        args['t'] = time.time()


        return render_template('part1_res.html', **args)


@app.route('/part2_res/', methods=['GET', 'POST'])
def part2_res():
    if request.method == 'GET':
        pass
    else:
        res = request.form['name']
        num = int(request.form['num'])
        from interface1 import app

        import time
        args = app(res, num)
        args['input'] = res
        args['t'] = time.time()

        return render_template('part2_res.html', **args)


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.run()
