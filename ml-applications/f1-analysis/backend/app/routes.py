from app import app

@app.route('/')
def index():
    return 'Welcome to the F1 Data Visualization Web App!'