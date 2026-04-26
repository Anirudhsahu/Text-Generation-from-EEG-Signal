import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import os
from predict import run_inference
import uuid
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'unmyeong'

DATABASE = 'database.db'

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone_number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone_number = request.form['phone_number']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        conn = get_db_connection()
        user_check = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if user_check:
            flash('Username or email already exists. Please choose a different one.', 'error')
            conn.close()
            return render_template('register.html')
        hashed_password = generate_password_hash(password)

        try:
            conn.execute(
                'INSERT INTO users (username, email, phone_number, password) VALUES (?, ?, ?, ?)',
                (username, email, phone_number, hashed_password)
            )
            conn.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An error occurred during registration. Please try again.', 'error')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/home')
def home():
    if 'username' not in session:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    user = conn.execute(
        'SELECT username, email, phone_number FROM users WHERE username = ?',
        (session['username'],)
    ).fetchone()
    conn.close()

    return render_template('home.html', user=user)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        if 'username' not in session:
            flash("Please log in first.", "error")
            return redirect(url_for('login'))

        file = request.files.get("file")

        if file and file.filename.endswith(".pkl"):
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(file_path)

            try:
                prediction = run_inference(file_path)

                conn = get_db_connection()
                conn.execute(
                    '''
                    INSERT INTO predictions (username, filename, result)
                    VALUES (?, ?, ?)
                    ''',
                    (session['username'], file.filename, str(prediction))
                )
                conn.commit()
                conn.close()

            except Exception as e:
                prediction = f"Error: {str(e)}"

        else:
            flash("Invalid file format. Only .pkl files allowed.", "error")

    return render_template('predict.html', prediction=prediction)

@app.route('/history')
def history():
    if 'username' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    records = conn.execute(
        'SELECT filename, result, created_at FROM predictions WHERE username = ? ORDER BY created_at DESC',
        (session['username'],)
    ).fetchall()
    conn.close()

    return render_template('history.html', records=records)

@app.route('/analytics')
def analytics():
    if 'username' not in session:
        flash("Please log in first.", "error")
        return redirect(url_for('login'))

    conn = get_db_connection()
    records = conn.execute(
        'SELECT result, created_at FROM predictions WHERE username = ?',
        (session['username'],)
    ).fetchall()
    conn.close()

    # Combine all text
    text = " ".join([r["result"] for r in records])

    # Clean text
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Better stopwords
    stopwords = set([
        'the','and','to','in','of','is','with','a','for','on',
        'this','that','was','as','by','it','from','at'
    ])

    words = [w for w in text.split() if w not in stopwords and len(w) > 3]
    clean_text = " ".join(words)

    # Generate WordCloud
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate(clean_text)

    # Save image
    image_path = os.path.join('static', 'wordcloud.png')
    wc.to_file(image_path)

    # --- Bar chart data ---
    from collections import Counter
    word_counts = Counter(words).most_common(10)

    labels = [w[0] for w in word_counts]
    values = [w[1] for w in word_counts]

    # --- Line chart (prediction length over time) ---
    dates = [r["created_at"] for r in records]
    lengths = [len(r["result"]) for r in records]

    return render_template(
        'analytics.html',
        labels=labels,
        values=values,
        dates=dates,
        lengths=lengths
    )


if __name__ == '__main__':
    app.run(debug=True)
