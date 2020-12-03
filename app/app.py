from flask import Flask, render_template, request, redirect
from plotchart import plotChart
from bokeh.embed import components
import predictP
from datetime import timedelta,datetime

app = Flask(__name__)

app.next_report_date,app.predicted_rate=predictP.get_UR_prediction()
app.next_prediction=datetime.strftime(datetime.strptime(app.next_report_date,"%Y-%m-%d")-timedelta(days=1),"%Y-%m-%d")

@app.route('/', methods=['POST', 'GET'])
def index():

    if request.method == 'GET':
        ticker='FB'
        
    else:
        ticker=request.form['ticker']
            
    plot = plotChart(ticker)
    script, div = components(plot)
    return render_template('index.html',script = script
                           , div = div
                           ,next_report_date=app.next_report_date
                           ,predicted_rate=app.predicted_rate
                          ,next_prediction=app.next_prediction)


if __name__ == '__main__':

    app.run(port=33507,debug=False)

