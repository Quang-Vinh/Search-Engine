import dictionary
import copy
import itertools
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


def expand_query(user_query, model):
    '''
    expands query using the rest of the methods in this library
    to get it back to a form usable by the retreival methods youill need to use: inter_model_2_boolean() or inter_model_2_vsm()
    '''
    query = get_inter_model(user_query)
    
    if len(query) > 1:
        query = expand_term_multiple(query, similarity_threshold=0)
    else:
        print('todo')
    
    if model == "boolean":
        return inter_model_2_boolean(query, user_query)
    elif model == "vsm":
        return inter_model_2_vsm(query)

def expand_term(term, similarity_threshold=0.5, include_hypernyms=False):
    '''
    expands term, returning a list containing itself and other terms in the wordnet with enough similarity
    '''
    expanded_term = [term]
    
    synonyms = wordnet.synset(term)
    for synonym in wordnet:
        print(synonym)
    
    return expanded_term
    
    
def expand_term_multiple(query, similarity_threshold=0.5, include_hypernyms=False):
    '''
    gets finds the most similar word senses and marks them for use in the expanded query if similarity is past the threshold
    uses path similarity as similarity measure, as well as vsm scoring
    '''
    
    for term_combo in itertools.combinations(query, 2):
        ab_sense_pair, ab_similarity = get_most_similar_synsets(term_combo[0][0], term_combo[1][0])
        if ab_similarity > similarity_threshold:
            term_a_synonyms = get_synonyms(ab_sense_pair[0])
            term_b_synonyms = get_synonyms(ab_sense_pair[1])
            for syn in term_a_synonyms:
                if not syn in [x[0] for x in term_combo[0][1]] and not syn in [x[0] for x in query]:
                    term_combo[0][1].append((syn, ab_similarity))
            for syn in term_a_synonyms:
                if not syn in [x[0] for x in term_combo[1][1]] and not syn in [x[0] for x in query]:
                    term_combo[1][1].append((syn, ab_similarity))
    return query

def get_synonyms(syns):
    '''
    gets synonyms from synset
    '''
    return [x.name() for x in syns.lemmas()]
    
def get_most_similar_synsets(term_1, term_2):
    '''
    takes in 2 terms, and finds a synset in each term's synsets with the greatest similarity
    '''
    term_1_syns = wordnet.synsets(term_1)
    term_2_syns = wordnet.synsets(term_2)
    max_similarity = 0
    
    for syn1 in term_1_syns:
        for syn2 in term_2_syns:
            similarity = syn1.path_similarity(syn2)
            if not similarity: similarity = 0
            if similarity > max_similarity:
                max_similarity = similarity
                best_pair = (syn1, syn2)
    return best_pair, max_similarity



def get_inter_model(query):
    '''
    takes a raw query string and returns a inter model representation of the query
    (the point is to have all the info available to work on using common methods and then spit it back out in a model specific form)
    '''

    query = query.replace('(', "")
    query = query.replace(')', "")
    query = query.replace('AND_NOT', "")
    query = query.replace('AND', "")
    query = query.replace('OR', "")
    
    inter_model_query = [(x, [(x, 1.0)]) for x in dictionary.tokenize(query)]
    return inter_model_query


def inter_model_2_boolean(query, raw_query_str):
    '''
    takes a query in intermodel representation and outputs the query in boolean form for the boolean model
    '''
    print(query)
    expanded_query = copy.deepcopy(raw_query_str)
    
    for syn_group in query:
        synonyms = [x[0] for x in syn_group[1]]
        boolean_syngroup_str = ' OR '.join(synonyms)
        expanded_query.replace(syn_group[0], boolean_syngroup_str)
    return expanded_query

def inter_model_2_vsm(query):
    '''
    takes a query in intermodel representation and outputs the query in boolean form for the boolean model
    '''
    expanded_query = []
    for synonym_groups in query:
        for j in synonym_groups[1]:
                expanded_query.append(j)
    return expanded_query
    
#test
print(get_most_similar_synsets("cat", "ocean"))

boolean_query = "money AND (cash OR coin)"
inter_model = get_inter_model(boolean_query)
print("result : ")
print(expand_query(boolean_query, "boolean"))

vector_query = "money cash coin"
inter_model = get_inter_model(vector_query)
print("result : ")
print(expand_query(vector_query, "vsm"))
