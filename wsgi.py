from app import create_app
from flask import jsonify

app = create_app('app.config.DevelopmentConfig')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message':'hello world'}), 200
#@app.route('/interview')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
