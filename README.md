# Question Auto-Completion
This is a question auto-completion system that uses a typed LSTM language model to generate completion predictions for a given question prefix.
The system is described in detail in [this thesis](http://ad-publications.informatik.uni-freiburg.de/theses/Master_Natalie_Prange_2019.pdf).

## Requirements
The QAC system requires approximately 11GB of RAM and takes about 4 minutes to load until operable.

## Run the API
All necessary datasets can be found under `/nfs/students/natalie-prange/` when using one of our chair's computer systems.

Use docker to run the QAC API on port 8181. You can build and run the docker container using the following commands:

    docker build -t qac .
    docker run -it -p 8181:80 -v /nfs/students/natalie-prange:/extern/data -v /nfs/students/natalie-prange/docker_output:/extern/output qac

Within the docker container, simply run

    python3 qac_api.py 80
    
to start the API. Alternatively, run `make help` to get additional information.

## Access the API
Once you've started the API on server `<host>` with port `<port>` (i.e. 8181 when using above command), you can access the API with

    http://<host>:<port>/?q=who%20played%20gand

This will return the top 5 completion predictions for the question prefix "who played gand" as JSON object in the format

    {
      "completions":[
        {
          "who played [sports team|q847326:Gandzasar-Kapan FC] ",
          "score":0.0023540909127749638
        },
        {
          "completion":"who played [fictional character|q177499:Gandalf] ",
          "score":0.001912843623237519
        },
        ...
      ]
    }

Wikidata entities are returned in the format `[<type>|<qid>:<label>]`.
