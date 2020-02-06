# Question Auto-Completion
This is a question auto-completion system that uses a typed LSTM language model to generate completion predictions for a given question prefix.
The system is described in detail in [this thesis](http://ad-publications.informatik.uni-freiburg.de/theses/Master_Natalie_Prange_2019.pdf).

## Requirements
The QAC system requires approximately 11GB of RAM and takes about 4 minutes to load until operable.

## Run the API
All necessary datasets can be found under `/nfs/students/natalie-prange/` when using one of our chair's computer systems.

Use docker to run the QAC API on port 8181. You can build the docker container and start the server using the following commands:

    docker build -t qac .
    docker run --rm -it -p 8181:80 -v /nfs/students/natalie-prange:/data qac

## Access the API
Once you've started the API on server `<host>` with port `<port>` (i.e. 8181 when using above command), you can access the API with

    http://<host>:<port>/?q=who%20played%20%5Bq177499%5D%20in%20lo

Wikidata entities in the question prefix must be in the format `[<QID>]`.
This will return the top 5 completion predictions for the question prefix _"who played [Gandalf] in lo"_ as JSON object in the format

    {
      "results":[
        {
          "completion":"who played [Gandalf] in [The Lord of the Rings: The Fellowship of the Ring] ",
          "matched_alias":"Lord of the Rings",
          "qids":["q177499","q127367"],
          "score":0.01866326314240102,
          "types":["fictional character","film"]
        },
        {
          "completion":"who played [Gandalf] in [The Lord of the Rings: The Return of the King] ",
          "matched_alias":"Lord of the Rings: The Return of the King",
          "qids":["q177499","q131074"],
          "score":0.0130516884507797,
          "types":["fictional character","film"]
        },
        ...
      ]
    }

- `completion` is the completion string with entities in the format `[<entity_label>]`.
- `matched_alias` is the alias which was matched against the current word prefix (_"lo"_).
If the current word prefix matches the entity label (e.g. if the current word prefix was _"the lo"_), `matched_alias` is the empty string `""`.
- `qids` is a list of QIDs of the entities in the completion string.
- `score` is the score computed for the completion.
- `types` is a list of types of the entities in the completion string.
