from flask import Flask, jsonify
from CVFiltration.cv_filtration import evaluate_cvs

app = Flask(__name__)

@app.route('/cv-filtration', methods=['POST'])
def filterCV():
    try:
        cv_scores = evaluate_cvs()
        print(cv_scores)
        return jsonify({"cvScores": cv_scores}), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
