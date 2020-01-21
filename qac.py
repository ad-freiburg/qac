# Copyright 2016-2020, University of Freiburg
# Author: Natalie Prange <prangen@informatik.uni-freiburg.de>


import logging
import re
import gensim
import string
import datrie
import math
import time
import unidecode
from operator import itemgetter
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import global_paths as gp
from lstm_lm import LM, PredictTypes

# Set up the logger
logging.basicConfig(format='%(asctime)s : %(message)s',
                    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

# Methods to be used for entity insertion
METHOD_COOCCURRENCE = "cooccurrence"
METHOD_WORD2VEC = "word2vec"
METHOD_SITELINKS = "fbscore"

# Index and mapping files
wd_path = gp.WIKIDATA_MAPPINGS
INDEX_FILE_PATH = wd_path + "total_file_norm_scores_wd_pure_v8_sepscores.txt"
INDEX_FILE_PATH_EV = wd_path+"total_file_norm_scores_wd_pure_v9_sepscores.txt"
TYPE_2_IDS_PATH = wd_path + "category_to_sorted_ids_wd_pure_v8.txt"
TYPE_2_IDS_PATH_EV = wd_path + "category_to_sorted_ids_wd_pure_v9_new.txt"
QID_TO_ALIASES_PATH = wd_path + "qid_to_aliases_wd_pure.txt"

# Language model files
LM_MODEL_FILE = gp.LANGUAGE_MODELS_LSTM + "model/model_100emb_512lu_025drop_512bs_15ep_wd_v8_comb"
LM_DATA_FILE = gp.LANGUAGE_MODELS_LSTM + "data/aq_gq_cw_combined_shuffled.ids"

# Word2Vec model files
w2v_path = gp.WORD2VEC_MODELS
WORD2VEC_PATH = w2v_path + "word2vec_entity_model_wd_200_5_20ep_lm_st_re"
WORD2VEC_PATH_EV = w2v_path + "word2vec_entity_model_wd_200_5_20ep_lm_st_re_train"

# Co-occurrence count files
CO_OCC_PATH = wd_path + "co-occurrence_pair_counts_ents_qid.txt"
CO_OCC_PATH_EV = wd_path+"co-occurrence_pair_counts_ents_qid_train.txt"
CO_OCC_WHICHTYPE_PATH = wd_path + "co-occurrence_pair_counts_ents_whichtype_qid.txt"
CO_OCC_WHICHTYPE_PATH_EV = wd_path + "co-occurrence_pair_counts_ents_whichtype_qid_train.txt"

# Flags adjust for evaluation purposes
INSERTION_METHOD = METHOD_COOCCURRENCE
APPEND_COMPLETE_ENTITIES = True
FILL_UP_WITH_W2V = INSERTION_METHOD == METHOD_COOCCURRENCE
HUMAN_PENALTY = True
CONS_ENTITY_PENALTY = True

# Categories for which not to use an alias but name roations
# e.g. Einstein Albert
individual_categories = ["Q5", "Q795052", "Q215627"]

# Misc
MAX_SUGGESTIONS = 10
DEFAULT_SUGGESTIONS = 5

# Captures entities in the format [<QID>]
ENTITY_PATTERN = r"\[([qQ][0-9]+?)\]"


def is_type(string):
    """Returns whether the given string is a type or not."""
    return (string.startswith("[") and string.endswith("]")
            and len(string) > 2)


def is_entity(string):
    """Returns whether the given string is an entity token or not."""
    return re.match(ENTITY_PATTERN, string) != None


def to_w2v_format(token):
    """Return a given entity token in the format as it would appear in the w2v
    model: I.e. lowercase and entities in the format "[<QID>]"

    Arguments:
    token - A string.

    >>> to_w2v_format("[Q123:test]")
    '[q123]'
    >>> to_w2v_format("[fictional character|Q123:test]")
    '[q123]'
    """
    token = token.lower()
    token = token.strip("[]")
    if "|" in token:
        token = token.split("|")[1]
    if ":" in token:
        # token is an entity. Return in w2v format: "[<qid>]"
        return "[" + token.split(":", 1)[0] + "]"
    else:
        return "[" + token + "]"


def get_rotations(name):
    """Get all rotations of a (multiword) name. Freebase-Easy additions in ()
    are being appended at the end of each rotation.

    Argument:
    name - the entity name.

    Example:
    >>> get_rotations("Samuel L Jackson")
    ['Samuel L Jackson', 'L Jackson Samuel', 'Jackson Samuel L']
    >>> get_rotations("Temple Grand (Biographical Film)")
    ['Temple Grand (Biographical Film)', 'Grand Temple (Biographical Film)']
    """
    suffix = re.findall(r"\([^\)]+\)", name)
    if suffix:
        suffix = suffix[-1]
        name = name.replace(suffix, "")
    name_lst = name.split()
    rotations = [name_lst[i:] + name_lst[:i] for i in range(len(name_lst))]
    if suffix:
        rotations = [r + [suffix] for r in rotations]
    return [' '.join(r) for r in rotations]


def tokenize_query(question, entity_pattern=ENTITY_PATTERN):
    """Gets a question prefix (string) and returns a list of tokens in which
    entities in the <entity_pattern> are not tokenized,

    Arguments:
    question - The question prefix. A string.

    >>> pattern = r"\[([^\]\[|]*?)\|([^\]\[|]*?)\]"
    >>> tokenize_query("What did [person|Karl Rove] say ?", pattern)
    ['what', 'did', '[person|Karl Rove]', 'say', '?']
    >>> tokenize_query("Who played [fictional_character|Gollum] in [film|The Lord of the Rings: The Two Towers] ?", pattern)
    ['who', 'played', '[fictional_character|Gollum]', 'in', '[film|The Lord of the Rings: The Two Towers]', '?']
    >>> tokenize_query("What is his [ [person|Charles Darwin] ] 's contribution to ontology ?", pattern)
    ['what', 'is', 'his', '[', '[person|Charles Darwin]', ']', "'s", 'contribution', 'to', 'ontology', '?']
    >>> tokenize_query("What role did [Q203047] play ?")
    ['what', 'role', 'did', '[Q203047]', 'play', '?']
    """
    tokenizer = RegexpTokenizer(r"""[^!"#$%&+;<=>@^\`{|}~“”\s_]+""")
    entities = list(re.finditer(entity_pattern, question))
    if len(entities) == 0:
        return tokenizer.tokenize(unidecode.unidecode(question.lower()))
    tokens = []
    # Save entities from cleaning process -> keep entity names as they are.
    for i in range(len(entities) + 1):
        if i == 0 and i < len(entities):
            q_part = question[0:entities[i].start(0)].lower()
            q_part = unidecode.unidecode(q_part)
            tokens += tokenizer.tokenize(q_part)
            tokens.append(entities[i].group(0))
        elif i < len(entities):
            q_part = question[entities[i-1].end(0):entities[i].start(0)]
            q_part = q_part.lower()
            q_part = unidecode.unidecode(q_part)
            tokens += tokenizer.tokenize(q_part)
            tokens.append(entities[i].group(0))
        else:
            q_part = question[entities[i-1].end(0):].lower()
            q_part = unidecode.unidecode(q_part)
            tokens += tokenizer.tokenize(q_part)
    return tokens


class CaseInsensitiveDict(dict):
    """ Careful: function for "in" operator is not implemented here so the key
    to be searched has to be converted to lower case first.
    """
    def __setitem__(self, key, value):
        super(CaseInsensitiveDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDict, self).__getitem__(key.lower())


class QAC:
    def __init__(self, lm_data_file=LM_DATA_FILE, lm_model_file=LM_MODEL_FILE,
                 evaluate=False):
        self.evaluate = evaluate

        # Mappings to ids as given in the INDEX_FILE
        self.type_to_ids = CaseInsensitiveDict()
        self.name_to_ids = CaseInsensitiveDict()
        self.qid_to_id = CaseInsensitiveDict()

        # The i-th entry of these lists is name, type or score, respectively,
        # for the entity with id i.
        self.qid_index = list()
        self.name_index = list()
        self.type_index = list()
        self.score_index = list()
        self.aliases_index = list()

        # Caches
        self.w2v_sim_cache = dict()
        self.w2v_nw_sim_cache = dict()
        self.lm_prob_cache = dict()
        self.lm_prob_hash_prefix = dict()
        self.context_prob_cache = dict()
        self.coocc_cache = dict()

        # Number of suggestions to be returned
        self.max_suggestions = DEFAULT_SUGGESTIONS

        # Initialize stopwords and lemmatizer
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.initialize_wordnet()

        # Get filenames
        if self.evaluate:
            coocc_file = CO_OCC_PATH_EV
            which_coocc_file = CO_OCC_WHICHTYPE_PATH_EV
            type_file = TYPE_2_IDS_PATH_EV
            w2v_path = WORD2VEC_PATH_EV
            index_file = INDEX_FILE_PATH_EV
        else:
            coocc_file = CO_OCC_PATH
            which_coocc_file = CO_OCC_WHICHTYPE_PATH
            type_file = TYPE_2_IDS_PATH
            w2v_path = WORD2VEC_PATH    
            index_file = INDEX_FILE_PATH

        # Load w2v model
        self.load_w2v_model(w2v_path)

        # Build indices
        self.build_indices(index_file)      

        # Build prefix trees
        self.build_type_trie(type_file)

        # Build type-2-ids mapping
        self.build_type_2_ids_mapping(type_file)

        # Build co-occurrence mappings
        self.co_occurrences = self.read_co_occurrences(coocc_file)
        whichtype_cooc = self.read_co_occurrences(which_coocc_file)
        self.co_occurrences.update(whichtype_cooc)

        # Build a prefix tree for the which-<type>'s
        self.whichtypes_trie = datrie.BaseTrie(self.all_chars)
        for whichtype in whichtype_cooc.keys():
            self.whichtypes_trie[whichtype] = 0

        # Initialize the LSTM language model
        self.initialize_lm(lm_data_file, lm_model_file)

        logger.info("Ready.")

    def load_w2v_model(self, w2v_path):
        """Load the word2vec model from the given file

        Arguments:
        w2v_path - path to the w2v model files
        """
        logger.info("Loading word2vec model from %s" % w2v_path)
        logging.getLogger('gensim').setLevel(logging.WARNING)
        w2v_model = gensim.models.KeyedVectors.load(w2v_path)
        self.word2vec = w2v_model

    def build_indices(self, index_file):
        """Create indices.

        Arguments:
        index_file - path to the index file
        """
        qid_to_aliases = dict()
        logger.info("Reading qid-2-alias mapping from file %s"
                    % QID_TO_ALIASES_PATH)
        with open(QID_TO_ALIASES_PATH, "r", encoding="utf8") as file:
            for line in file:
                lst = line.strip().split("\t")
                qid = lst[0]
                aliases = [unidecode.unidecode(a) for a in lst[1:]]
                aliases = [a for a in aliases if len(a) > 1
                           and a[0] != "?"]
                qid_to_aliases[qid] = aliases

        weird_chars = ""
        logger.info("Reading indices from file %s" % index_file)
        with open(index_file, "r", encoding="utf8") as file:
            for line in file:
                lst = line.split("\t")
                score_prim = float(lst[3])
                score_sec = float(lst[4])
                id = int(lst[0])
                qid, name = lst[1].split(":", 1)
                # Remove accents etc to keep the prefix trie small enough to
                # not yield a seg fault
                noaccent_name = unidecode.unidecode(name)
                typ = lst[2]
                prim, sec = typ.split("/", 1)
                qid_prim = prim.split(":", 1)[0]
                qid_sec = sec.split(":", 1)[0]
                typ = typ.lower()
                self.qid_index.append(qid.lower())
                self.qid_to_id[qid] = id
                self.name_index.append(name)
                self.type_index.append(typ)
                self.score_index.append((prim.lower(), score_prim, sec.lower(),
                                        score_sec))
                if qid_prim in individual_categories or \
                        qid_sec in individual_categories:
                    # Alias count when using only rotations: 1,501,895
                    self.aliases_index.append(get_rotations(noaccent_name))
                else:
                    # Alias count when using aliases and rotations only for
                    # individuals: 1,330,988
                    aliases = [name] + qid_to_aliases.get(qid, [])
                    self.aliases_index.append(aliases)
                if noaccent_name.lower() not in self.name_to_ids:
                    self.name_to_ids[noaccent_name] = list()
                self.name_to_ids[noaccent_name].append(id)
                if typ == "" or typ == "unknown":
                    continue
                for ch in noaccent_name:
                    if ch not in string.printable and ch not in weird_chars:
                        weird_chars += ch
        self.all_chars = string.printable + weird_chars

    def build_type_trie(self, type_file):
        """Build a prefix tree for each type containing all its assigned
        entity names and aliases

        Arguments:
        type_file - the path to the type_2_ids mapping
        """
        # Create type_to_ids dict by reading from file type_to_sorted_ids.
        # The file contains only ids with a score of at least x
        self.type_to_trie = CaseInsensitiveDict()
        logger.info("Building type tries from file %s" % type_file)
        with open(type_file, "r", encoding="utf8") as file:
            for line in file:
                lst = line.split("\t")
                lst[-1] = lst[-1].strip("\n")
                typs = lst[0].lower().split("/")
                ids = lst[1:]
                ids = [int(id) for id in ids]
                for typ in typs:
                    self.type_to_trie[typ] = datrie.BaseTrie(self.all_chars)
                    for id in ids:
                        for al in self.aliases_index[id]:
                            al = al.lower()
                            # Append ID to prefix to make it unique
                            if (al in self.type_to_trie[typ]
                                    and self.type_to_trie[typ][al] != id):
                                al += " " + str(id)
                            self.type_to_trie[typ][al] = id

                    self.type_to_ids[typ] = ids

    def build_type_2_ids_mapping(self, type_file):
        """Build the type 2 ids mapping

        Arguments:
        type_file - the path to the type_2_ids mapping
        """
        logger.info("Building type-2-id mapping from file %s" % type_file)
        with open(type_file, "r", encoding="utf8") as file:
            for line in file:
                lst = line.split("\t")
                lst[-1] = lst[-1].strip("\n")
                typs = lst[0].lower().split("/")
                ids = lst[1:]
                ids = [int(id) for id in ids]
                for typ in typs:
                    self.type_to_ids[typ] = ids


    def read_co_occurrences(self, coocc_file, min_score=0.001):
        """Read co-occurrences from the given

        Arguments:
        coocc_file - File containing word pairs and their co-occurrence score
        min_score - Minimum cooccurrence score to include word pair in mapping
        """
        logger.info("Reading co_occurrences from %s" % coocc_file)
        co_occurrences = dict()
        with open(coocc_file, "r", encoding="utf8") as file:
            for line in file:
                qid1, qid2, score = line.strip().split("\t")
                score = float(score)
                if score >= min_score:
                    if qid1 not in co_occurrences:
                        co_occurrences[qid1] = dict()
                    co_occurrences[qid1][qid2] = score
        return co_occurrences

    def initialize_lm(self, lm_data_file, lm_model_file):
        """Initialize the language model with the given data and model files.

        Arguments:
        lm_data_file - File that contains the id sequences or lm vocab
        lm_model_file - Model file containing trained weights
        """
        logger.info("Initializing language model with data file %s"
                    " and model file %s" % (lm_data_file, lm_model_file))
        self.lm = LM(lm_data_file)
        self.lm.load_model(lm_model_file)
        self.lm.initialize_trie()

    def initialize_wordnet(self):
        """Make sure the WordNet corpus is loaded by calling lemmatize().
        This prevents long loading times when lemmatize() is called during
        completion for the first time.
        """
        logger.info("Initializing WordNet")
        self.lemmatizer.lemmatize("initialize")

    def set_max_suggestions(self, max_suggestions=DEFAULT_SUGGESTIONS):
        """Set the number of completions to present to the user.

        Arguments:
        max_suggestions - Max number of possible completion suggestions
        """
        self.max_suggestions = max_suggestions
        if self.max_suggestions > MAX_SUGGESTIONS:
            self.max_suggestions = MAX_SUGGESTIONS
        if self.max_suggestions <= 0:
            self.max_suggestions = 1

    def clear_caches(self, max_size):
        """Clear the caches.

        Arguments:
        max_size - if True only clear the cache if its size is bigger than the
                   corresponding size in max_sizes.
        """
        max_sizes = [10000, 10000, 100, 100, 100]
        caches = [self.w2v_sim_cache, self.w2v_nw_sim_cache,
                  self.lm_prob_cache, self.lm_prob_hash_prefix,
                  self.context_prob_cache]
        for i, cache in enumerate(caches):
            if max_size:
                if len(cache) > max_sizes[i]:
                    logger.info("Cleared cache %d" % i)
                    cache.clear()
            else:
                logger.info("Cleared cache %d" % i)
                cache.clear()

    def complete_question(self, question_prefix):
        """Returns the most likely completions for a given question prefix and
        their computed scores.

        Arguments:
        question_prefix - The typed question so far. Entities as [<QID>]
        """
        # Store whether the last character was a space
        space_last = len(question_prefix) > 0 and question_prefix[-1] == " "

        # Tokenize question prefix
        prefix_toks = tokenize_query(question_prefix)
        logger.info("tokenized query: %s" % prefix_toks)

        # Clear cache if the input field is empty or just one word
        if len(prefix_toks) <= 1:
            self.clear_caches(max_size=False)

        # Clear caches if they are becoming too big
        self.clear_caches(max_size=True)

        # Get completion predictions
        start = time.time()
        completions = self.get_completions(prefix_toks, space_last)
        logger.info("Time to get final words:\t%f" % (time.time() - start))

        if completions == []:
            return []

        # Get qids, names and primary types of entities in current question prefix
        prefix_qids = []
        named_prefix_toks = []
        prefix_types = []
        for t in prefix_toks:
            if is_entity(t):
                qid = t.strip("[]").lower()
                id = self.qid_to_id[qid]
                name = self.name_index[id]
                typ = self.type_index[id].split("/")[0].split(":")[1]
                prefix_qids.append(qid)
                named_prefix_toks.append("[" + name + "]")
                prefix_types.append(typ)
            else:
                named_prefix_toks.append(t)

        # Put the predicted question (prefix) together
        results = []
        for word, qid, alias, typ, score, ind in completions[:self.max_suggestions]:
            # Get index at which the completion should be appended
            prefix_start = -ind if ind != 0 else None
            # Get question prefix to which completion should be appended
            question_prefix = ' '.join(named_prefix_toks[:prefix_start])
            if len(named_prefix_toks) > 1 or space_last:
                question_prefix += " "

            # Append the completion
            question_prefix += word

            # Append a space if the question is not terminated by "?"
            if len(question_prefix) > 0 and not question_prefix[-1] == "?":
                question_prefix += " "

            question_qids = prefix_qids + [qid]
            question_types = prefix_types + [typ]
            results.append((question_prefix, question_qids, question_types, alias, score))

        return results

    def get_completions(self, context, space_last=True):
        """Given a list of words predict which word is most likely to follow.

        Arguments:
        context - list of strings, the words typed so far
        space_last - indicates whether the last word is complete or a fragment
        """
        # Get the index of the last entity in the context
        ent_indices = [i for i, w in enumerate(context)
                       if re.match(ENTITY_PATTERN, w)]
        # Get a list of previous context entities
        context_ents = [to_w2v_format(c) for c in re.findall(ENTITY_PATTERN,
                        ' '.join(context))]
        context_ents = [c for c in context_ents if c in self.word2vec.wv.vocab]

        # Get plain words and types as they appear in the lm
        lm_context = self.get_lm_context(context)

        # A prefix can start at the beginning of the context or after the last
        # entity if there is one
        poss_start = 0 if len(ent_indices) == 0 else ent_indices[-1] + 1

        # Get completion predictions
        result = self.get_best_words(poss_start, lm_context, space_last,
                                     context_ents, INSERTION_METHOD)
        best_words, complete_entities, applied_method = result
        best_words.sort(key=itemgetter(2, 0), reverse=True)

        # Fill up the completion predictions with w2v based completions if not
        # enough completions were predicted so far
        if FILL_UP_WITH_W2V and applied_method == METHOD_COOCCURRENCE and \
                len(best_words) < self.max_suggestions:
            logger.info("Fill up predictions with word2vec predictions")
            best_words_w2v, _, _ = self.get_best_words(poss_start,
                                                       lm_context,
                                                       space_last,
                                                       context_ents,
                                                       METHOD_WORD2VEC)
            best_words_w2v.sort(key=itemgetter(2, 0), reverse=True)
            best_words += best_words_w2v

        # Remove entities that occur several times in completion predictions
        best_words = self.remove_double_suggestions(best_words)

        # Insert completely typed entities into completion predictions
        if APPEND_COMPLETE_ENTITIES:
            best_words = self.insert_complete_entities(complete_entities,
                                                       best_words)

        # Put together final completion suggestions
        return self.format_final_predictions(best_words, lm_context)
        
    def get_best_words(self, poss_start, lm_context, space_last, context_ents,
                       method):
        """Get the best words and entities for every possible prefix
        """
        orig_method = method
        best_words = []
        complete_entities = []
        max_index = len(lm_context)+(space_last or len(lm_context) == 0)
        context_p = 1
        for prefix_start in range(poss_start, max_index):
            # Get prefix string
            if prefix_start == len(lm_context) or len(lm_context) == 0:
                prefix = ""
            else:
                prefix = ' '.join(lm_context[prefix_start:]) + ' '*space_last

            # Check if the word before the current word prefix is an entity
            is_consecutive_entity = (lm_context and prefix_start != 0 and \
                                    is_type(lm_context[prefix_start-1]))

            # Get words and types suggested by the language model
            sorted_words = self.get_lm_predictions(prefix, lm_context,
                                                   prefix_start, space_last)

            # Compute the probability of the context and multiply it with the
            # probability of the predicted word to make sure, predictions after
            # uncommon words such as "harry" don't have a higher probability
            # (due to fewer predictions) than the entity "harry potter"
            hash_context = tuple(lm_context[:prefix_start])
            if hash_context in self.context_prob_cache:
                # Retrieve context probability from cache
                new_context_p = self.context_prob_cache[hash_context]
            else:
                # Compute context probability and add to cache
                rel_context = lm_context[:prefix_start]
                new_context_p = self.lm.probability_for_context(rel_context)
                self.context_prob_cache[hash_context] = new_context_p
            context_p *= new_context_p
            # Make the difference between low probs and high ones smaller
            # --> don't prefer small contexts as much
            context_p_factor = math.log(context_p * 100 + 0.1, 10) + 1
            logger.debug("Context probability factor: %f" % context_p_factor)

            # Get contexts
            coocc_which, w2v_which = self.get_which_type_contexts(lm_context,
                                                                  prefix_start)
            coocc_context = [w.strip("[]") for w in context_ents + coocc_which
                             if w.strip("[]") in self.co_occurrences]
            w2v_context = context_ents + w2v_which

            # Get the normal word w2v context
            nw_w2v_context = [self.lemmatizer.lemmatize(w)
                              for w in lm_context[:prefix_start]
                              if w not in self.stopwords
                              and self.lemmatizer.lemmatize(w)
                              in self.word2vec.wv.vocab
                              and not w.startswith("[")
                              and not w.startswith("<")] + context_ents \
                                                         + w2v_which

            # Determine the method with which to score the entites
            if coocc_context != [] and orig_method == METHOD_COOCCURRENCE:
                method = METHOD_COOCCURRENCE
                context_words = coocc_context
            elif (FILL_UP_WITH_W2V or orig_method == METHOD_WORD2VEC) and \
                    w2v_context != [] and orig_method != METHOD_SITELINKS:
                method = METHOD_WORD2VEC
                context_words = w2v_context
            else:
                method = METHOD_SITELINKS
                context_words = []
            logger.info("Context words: %s" % context_words)
            logger.info("Scoring entities by method '%s'" % method)

            # Score words and fill in entities for abstract types.
            best_words_for_index = []
            count = 0
            qid_set = set()
            for word, entity, ngram_p, index in sorted_words:
                if len(best_words_for_index) > self.max_suggestions**2:
                    break
                if is_type(word):
                    primary, secondary = word.strip("[]").split("/")
                    primary = primary.replace("_", ":", 1).replace("_", " ")
                    best_entities = self.predict_entity(primary, prefix,
                                                        context_words, method)
                    # Compute the final score for each word
                    for qid, s in best_entities:
                        final_score = self.get_score(ngram_p*context_p_factor,
                                                     s,
                                                     context_words == [])

                        # Penalize consecutive entities
                        if is_consecutive_entity and CONS_ENTITY_PENALTY:
                            final_score *= 0.04

                        # FOR EVALUATION: for testing sitelinks method balance
                        # scoring of normal words and entities. Without the
                        # penalty, entities have a higher preference compared
                        # to the co-occurrence entity scoring method
                        if coocc_context != [] and method == METHOD_SITELINKS:
                            final_score *= 0.2

                        # Get types of entity. Since I'm only querying for prim
                        # class right now, this is needed, since otherwise ents
                        # are shown with the wrong primary class. Only a matter
                        # of appearance though since I'm querying for the type
                        # of each entity when parsing the question prefix
                        id = self.qid_to_id[qid]
                        typ = "[" + self.type_index[id] + "]"

                        # Penalize overrepresented type "Q5:human"
                        if typ.startswith("[q5:human") and HUMAN_PENALTY:
                            final_score *= 0.2

                        # Make sure no entity is added twice for the same index
                        if qid not in qid_set:
                            best_words_for_index.append((typ, qid,
                                                         final_score, index))
                            qid_set.add(qid)

                else:
                    s = None
                    if len(context_ents + w2v_which) != 0:
                        try:
                            hash_w2v_context = (word, tuple(nw_w2v_context))
                            if hash_w2v_context in self.w2v_nw_sim_cache:
                                # Retrieve similarity from cache
                                sim = self.w2v_nw_sim_cache[hash_w2v_context]
                            else:
                                # Compute similarity and add to cache
                                ctxt = nw_w2v_context
                                sim = self.word2vec.wv.n_similarity([word],
                                                                    ctxt)
                                self.w2v_nw_sim_cache[hash_w2v_context] = sim

                            if sim > 0:
                                s = sim * 0.1
                            else:
                                s = 0.0001 * 0.1
                        except KeyError as e:
                            s = 0.0001 * 0.1
                            logger.debug("Error: %s not in vocabulary." % e)
                    final_score = self.get_score(ngram_p*context_p_factor,
                                                 s,
                                                 context_words == [])
                    best_words_for_index.append((word, "", final_score, index))
                    count += 1
            best_words_for_index.sort(key=itemgetter(2, 0), reverse=True)
            logger.debug("Best words for index: %s" % best_words_for_index)

            # Add completion predictions for the current word prefix to the
            # overall completion predictions
            best_words += best_words_for_index[:self.max_suggestions]

            # Append entity to completions, if its name is completely typed.
            prefix_len = len(lm_context[prefix_start:])
            complete_entities += self.get_complete_entity(prefix, prefix_len)

        return best_words, complete_entities, method

    def get_lm_predictions(self, prefix, context, prefix_start, space_last):
        """Get words suggested by the language model given a prefix and a
        context.

        Arguments:
        prefix - A string.
        context - The tokenized query. A list of strings.
        prefix_start - Start index of the prefix
        space_last - A boolean indicating whether a space was inserted last.
        """
        predict_types = PredictTypes.ALSO
        if prefix_start < len(context) - 1 or (prefix_start == len(context) - 1
                                               and space_last):
            # if several words form the last prefix, don't predict normal words
            predict_types = PredictTypes.ONLY

        # Returns the possible words sorted by score
        logger.info("Get predictions for context %s and prefix '%s'" %
                    (context[:prefix_start], prefix))
        hash_context = (tuple(context[:prefix_start]), predict_types)
        if hash_context in self.lm_prob_cache \
                and prefix.startswith(self.lm_prob_hash_prefix[hash_context]):
            # Retrieve lm probabilities from cache and only keep those that
            # match the current prefix
            pred_words = self.lm_prob_cache[hash_context]
            pred_words = [(w, p) for w, p in pred_words if w.startswith(prefix)
                          or is_type(w)]
        else:
            # Compute lm probabilities
            pred_words = self.lm.predict_words(context[:prefix_start],
                                               prefix=prefix,
                                               predict_types=predict_types,
                                               max_words=1000000)
            self.lm_prob_cache[hash_context] = pred_words
            self.lm_prob_hash_prefix[hash_context] = prefix

        prefix_len = len(context[prefix_start:])
        pred_words = [(w, "", p, prefix_len) for w, p in pred_words[:30]
                      if w not in ["_UNK_", "[unknown]", "["]]
        logger.debug("predicted words %s etc." % pred_words)

        return pred_words

    def predict_entity(self, typ, prefix, context_words, method):
        """Predict an entity for given prefix, previous entities or previous
        words.

        Arguments:
        typ - type of entity, as occuring in processed_questions.txt without []
        prefix - prefix of the word which will be predicted.
        context_words - list of w2v relevant tokens (eg non-stopwords+entities)
        """
        max_words = []
        if typ not in self.type_to_ids:
            return max_words

        # Get possible entities (by index file score) for prefix and type
        poss_entities = []
        if prefix:
            added_ids = set()
            for id in sorted(self.type_to_trie[typ].values(prefix))[:500]:
                # Avoid taking the same entity twice due to aliases
                if id not in added_ids:
                    qid = self.qid_index[id]
                    name = unidecode.unidecode(self.name_index[id]).lower()
                    score = self.get_score_for_typ(id, typ)
                    if not name.startswith(prefix):
                        # Penalize entities if the prefix only matches their
                        # alias but not their label
                        score *= 0.6
                    elif name == prefix:
                        # Prefer entities that are completely typed
                        score = 1
                    poss_entities.append((qid, score))
                    added_ids.add(id)
        else:
            for id in self.type_to_ids[typ][:1000]:
                qid = self.qid_index[id]
                name = self.name_index[id]
                score = self.get_score_for_typ(id, typ)
                poss_entities.append((qid, score))

        # Compute the scores for the matches using the specified method
        if method == METHOD_SITELINKS or context_words == []:
            # Compute an entity's score based on its sitelink count
            if prefix:
                poss_entities.sort(key=itemgetter(1), reverse=True)
            if len(poss_entities) > 0:
                max_words = poss_entities[:self.max_suggestions]

        elif method == METHOD_COOCCURRENCE:
            # Compute an entity's score based on co-occurrence counts
            poss_coocc_qids = defaultdict(float)
            # Get the cooccurrence for each word in the context
            for context in context_words:
                for qid, score in self.co_occurrences[context].items():
                    poss_coocc_qids[qid] += score
            # Compute the co-occurrence score for each candidate entity
            for qid, s in poss_entities:
                if qid in poss_coocc_qids:
                    # Take the mean of the co-occurrence scores
                    similarity = poss_coocc_qids[qid] / len(context_words)
                    max_words.append((qid, similarity))
            max_words.sort(key=itemgetter(1), reverse=True)
            max_words = max_words[:self.max_suggestions]
            logger.debug("Co-occurrence words for %s: %s" % (typ, max_words))

        elif method == METHOD_WORD2VEC:
            # Compute an entity's score based on w2v similarity
            for qid, s in poss_entities:
                w2v_entity = to_w2v_format(qid)
                # Avoid s.th. like:
                # "why did [person|tom cruise] marry [person|tom cruise]"
                if w2v_entity in context_words:
                    continue
                try:
                    hash_w2v_context = (w2v_entity, tuple(context_words))
                    if hash_w2v_context in self.w2v_sim_cache:
                        # Retrieve w2v similarity from cache
                        similarity = self.w2v_sim_cache[hash_w2v_context]
                    else:
                        # Compute w2v similarity
                        w2v_ent = [w2v_entity]
                        ctxt = context_words
                        similarity = self.word2vec.wv.n_similarity(w2v_ent,
                                                                   ctxt)
                        # Add w2v similarity to cache
                        self.w2v_sim_cache[hash_w2v_context] = similarity

                    if similarity > 0:
                        similarity *= (math.log(s+1) - (math.log(2))+1)
                        if INSERTION_METHOD == METHOD_WORD2VEC:
                            similarity *= 0.01
                        max_words.append((qid, similarity))
                except KeyError as e:
                    logger.debug("Error: %s not in vocabulary." % e)
            max_words.sort(key=itemgetter(1), reverse=True)
            max_words = max_words[:self.max_suggestions]

        return max_words

    def get_complete_entity(self, prefix, index):
        """Returns the entities that are typed completely by the given prefix.
        Completely typed entities are returned in a list.

        Arguments:
        prefix - A string.
        index - The list-length of the current word prefix.
        """
        complete_entities = []
        if len(prefix) > 3 and prefix in self.name_to_ids:
            ids = self.name_to_ids[prefix]
            for id in ids:
                typ = self.type_index[id]
                qid = self.qid_index[id]
                complete_entities.append((typ, qid, 1, index))
        return complete_entities

    def insert_complete_entities(self, complete_entities, best_words):
        """Insert completely typed entities into the predicted completions.

        Arguments:
        complete_entities - list of completely typed entities
        best_words - list of predicted completions
        """
        # Only insert completely typed entities that do not appear yet
        insertion_words = []
        for el in complete_entities:
            entity_name = el[1]
            entity_index = el[3]
            index = self.max_suggestions - len(complete_entities) + 1
            ent_ind_lst = [(e, i) for _, e, _, i in best_words[:index]]
            if (entity_name, entity_index) not in ent_ind_lst:
                insertion_words.append(el)
        # Insert complete entities into completion predictions presented to the
        # user
        max_ins_index = max(0, self.max_suggestions - len(insertion_words))
        for i, word in enumerate(insertion_words):
            ins_index = max_ins_index + i
            best_words.insert(ins_index, word)
        return best_words

    def remove_double_suggestions(self, prediction_list):
        """Remove entities from the prediction list, that are also predicted
        for a higher index. I.e. avoid prediction lists like:
            "Who wrote the [Written_Work|The Bible]"
            "Who wrote [Written_Work|The Bible]"
        by only suggesting the latter.

        Arguments:
        prediction_list - list of prediction tuples
        """
        index1 = 0
        while index1 < len(prediction_list):
            w, e, s, i = prediction_list[index1]
            if index1+1 < len(prediction_list):
                index2 = index1 + 1
                while index2 < len(prediction_list):
                    w2, e2, s2, i2 = prediction_list[index2]
                    if (e == e2 and e != "") or (e == "" and e2 == ""
                                                 and w == w2):
                        logger.info("Remove %s" % ([w2, e2, s2, i2]))
                        if i2 < i or (i2 == i and s2 <= s):
                            del prediction_list[index2]
                        else:
                            del prediction_list[index1]
                            index1 -= 1
                            break
                    index2 += 1
            index1 += 1
        return prediction_list

    def format_final_predictions(self, completions, lm_context):
        """Put the final completion predictions into the correct format to
        have them ready to be appended to the current question prefix.

        Arguments:
        completions - list of completion predictions
        lm_context - the language model context. Needed to determine whether an
                     entity label matches the current word prefix or just its
                     alias
        """
        final_completions = []
        for w, q, s, i in completions[:self.max_suggestions]:
            word = w
            alias = ""
            typ = ""
            if q:
                id = self.qid_to_id[q]
                name = self.name_index[id]
                # Check if the completion was done for an alias of the entity
                norm_name = unidecode.unidecode(name).lower()
                prefix = ' '.join(lm_context[len(lm_context)-i:])
                prim_class = w.strip("[]").split("/")[0]
                _, typ = prim_class.split(":", 1)
                if not norm_name.startswith(prefix):
                    for al in self.aliases_index[id]:
                        norm_al = unidecode.unidecode(al).lower()
                        if norm_al.startswith(prefix):
                            alias = al
                            break
                word = "[" + name + "]"
            final_completions.append((word, q, alias, typ, s, i))
        return final_completions

    def get_score_for_typ(self, id, typ):
        """ Separated scores for primary and secondary class --> take the score
        for which the current type matches.

        Arguments:
        id - the id of the current entity
        typ - the type for which the score should be retrieved (primary or
              secondary)
        """
        scores = self.score_index[id]
        score = 0
        if scores[0] == typ:
            score = scores[1]
        elif scores[2] == typ:
            score = scores[3]
        else:
            logger.error("Something went wrong with the scores: score_typ: %s"
                         % scores[0])
        return score

    def get_score(self, lm_prob, score, use_sitelinks):
        """Computes the final score for a word. This score depends on whether
        it's a normal word or an entity and whether the score so far is a w2v
        similarity or a freebase score.

        Arguments:
        lm_prob - probability of this word/type according to the ngram-model
        score - the score of the word. Either a w2v similarity or a freebase
                score. If score is None the word is a normal word.
        use_sitelinks - whether the scores are pure sitelinks scores
        """
        if not score:
            score = 0.01 if use_sitelinks else 0.01

        exponent = 0.3
        return lm_prob * score ** exponent

    def get_lm_context(self, context):
        """Takes a set of context words and entity mentions and transforms them
        into a context as it would appear in the language model with entities
        replaced by their type.

        Arguments:
        context - a list of strings, the words and entities typed so far
        """
        lm_context = []
        for c in context:
            if is_type(c):
                qid = to_w2v_format(c).strip("[]")
                if qid in self.qid_to_id:
                    typ = self.type_index[self.qid_to_id[qid]]
                else:
                    typ = "unknown:unknown"
                typ = typ.lower().replace(" ", "_").replace(":", "_")
                lm_context.append("[" + typ + "]")
            else:
                lm_context.append(c)
        return lm_context

    def get_which_type_contexts(self, lm_context, prefix_start):
        """Get the which <type> token as context for entity insertion for both
        methods, co-occurrence and w2v.

        Arguments:
        lm_context - the language model context. A list of tokens.
        prefix_start - the start of the current word prefix
        """
        coocc_which = []
        w2v_which = []
        if len(lm_context[:prefix_start]) > 1 and lm_context[0] == "which" \
                and not is_type(lm_context[1]):
            w = lm_context[1]
            # Check if the which-type stretches over multiple words and add all
            # of these words
            if len(lm_context) > 2:
                lem_w = self.lemmatizer.lemmatize(w)
                for typ in self.whichtypes_trie.keys(lem_w):
                    lem_lm_context = [self.lemmatizer.lemmatize(tok)
                                      for tok in lm_context]
                    q_prefix = ' '.join(lem_lm_context[1:])
                    if q_prefix.startswith(typ):
                        coocc_which.append(typ)
            if coocc_which:
                w2v_which = coocc_which[0].split()
            else:
                w = self.lemmatizer.lemmatize(w)
                if w in self.word2vec.wv.vocab:
                    w2v_which.append(w)
        return coocc_which, w2v_which

    def set_log_level(self, level):
        logger.setLevel(level)
