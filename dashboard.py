from flask import Flask, render_template,send_from_directory, request, redirect, url_for, session, jsonify
import os
import pandas as pd
from datetime import datetime


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management
app.config['SESSION_PERMANENT'] = False

csv_file_path = './logs/logs.csv'

# Set the directory where your logs are located
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

@app.route('/logs/<path:filename>')
def custom_logs(filename):
    return send_from_directory(LOGS_DIR, filename)


def get_file_modification_time():
    return os.path.getmtime(csv_file_path)


@app.before_first_request
def clear_session_on_start():
    session.clear()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # TODO: Replace the hardcoded username and password
        if username == 'Tahaluf' and password == '123':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid Credentials. Please try again.')
    return render_template('login.html')

@app.route('/')
def root():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template('home.html')

@app.route('/view-todays-attendance')
def view_todays_attendance():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    today = datetime.now()
    date = today.strftime('''%d-%m-%Y''')
    file_name = f"./logs/{date}.csv"

    if os.path.exists(file_name):
        data = pd.read_csv(file_name, header=None)
        attendance_records = []
        for _, row in data.iterrows():
            record = {
                'student_image': "https://cdn-icons-png.flaticon.com/512/2302/2302834.png",
                'student_name': row[1],
                'student_class': "4B",
                'time_of_entrance': row[0],
                'camera': row[2],
                'logging_image': row[3]  # Assuming the path to the logging image is the last element
            }
            attendance_records.append(record)
    else:
        attendance_records = []

    return render_template('view_todays_attendance.html', attendance_records=attendance_records)


@app.route('/file-mod-time')
def file_mod_time():
    return jsonify({'mod_time': get_file_modification_time()})

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
