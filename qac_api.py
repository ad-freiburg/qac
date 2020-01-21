# Copyright 2020, University of Freiburg
# Author: Natalie Prange <prangen@informatik.uni-freiburg.de>


import flask
import sys
from flask import request, jsonify
from urllib import parse
from qac import QAC

app = flask.Flask(__name__)

# Initialize the QAC system
qac = QAC()


@app.route('/', methods=['GET'])
def qac_api():
    question_prefix = request.args.get('q', '')
    # Normalize question.
    question_prefix = parse.unquote(question_prefix)

    # Send results with proper HTTP headers.
    results = qac.complete_question(question_prefix)
    json_obj = [{"completion": compl,
                 "qids": qids,
                 "types": types,
                 "matched_alias": alias,
                 "score": score}
                for compl, qids, types, alias, score in results]
    json_obj = {"results": json_obj}
    return jsonify(json_obj)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        port = 8183
    elif len(sys.argv) == 2:
        port = int(sys.argv[1])
    else:
        print("Usage: python3 %s <port>" % sys.argv[0])
        exit(1)

    app.run(debug=False, host="::", port=port)
