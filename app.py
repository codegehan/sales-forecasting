from flask import Flask, render_template, request
import io
import base64
from forecast import read_and_predict_sales
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file and file.filename.endswith('.csv'):
            # Read file data
            file_stream = io.StringIO(file.stream.read().decode("UTF8"))
            
            # Get predictions
            historical, predictions = read_and_predict_sales(file_stream)
            
            if historical is None:
                return render_template('index.html', error="Error processing file")

            # Generate plot
            img = io.BytesIO()
            plt.figure(figsize=(10, 6))
            plt.plot(historical['date'], historical['sales'], label='Historical Sales', marker='o')
            plt.plot(predictions['date'], predictions['predicted_sales'], label='Predicted Sales', marker='o', linestyle='--')
            plt.title('Sales Forecast')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')

            # Prepare data for tables
            historical_html = historical[['date', 'sales']].to_html(index=False, classes='table table-striped')
            predictions_html = predictions.to_html(index=False, classes='table table-striped')
            
            return render_template('index.html', 
                                 historical_table=historical_html,
                                 predictions_table=predictions_html,
                                 plot_url=plot_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)