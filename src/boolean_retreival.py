import re
import pickle
import dictionary
import wildcard_management

def get_vinhs_garbage_pickle_thing():
    return pickle.load(open('../indexes/UofO_Courses_index.pkl', 'rb'))

def resolve_single_term(term, index):
    #returns the docIDs for a single search term
    term = index.dictionary.preprocess_document(term)
    if term[0].find('*') >= 0:
        return resolve_wildcard_term(term[0], index)
    else:
        result = index.get_postings(term[0])
        if not result:
            result = set()
        return result

def resolve_wildcard_term(term, index):
    #intersecs all terms in the dictionary matching up with the wildcard
    matching_words = wildcard_management.get_indexed_words(term)
    results = set()
    for word in matching_words:
        results = results.union(resolve_single_term(word, index))
    return results

def get_op_parse(query_string):
    #theres got to be a better way but im legit too dumb
    level = 0
    pre_arg = ''
    post_arg = ''
    op_arg = ''
    for parsed_op in re.finditer('\s(AND|OR|AND_NOT)\s', query_string):
        pre_arg = query_string[0 : parsed_op.span()[0]]
        post_arg = query_string[parsed_op.span()[1] : ]
        op_arg = parsed_op.group(1)

        if is_proper_parentheses(pre_arg) and is_proper_parentheses(post_arg):
            return (pre_arg, op_arg, post_arg)
    return None

def is_proper_parentheses(expr):
    #checks if the input expression has properly closed brackets
    level = 0
    for i in expr:
        if i == '(':
            level += 1
        elif i == ')':
            level -= 1
    return level == 0

def handle_operation(set1, set2, operation):
    if operation == 'AND':
        return set(set1).intersection(set(set2))
    elif operation == 'OR':
        return set(set1).union(set(set2))
    elif operation == 'AND_NOT':
        return set(set1) - set(set2)

def recursive_parse(query_string, index):
    #re.match is left to right right?
    #yeah its left to right
    result  = {}

    if re.fullmatch('\(.+\)', query_string):
        result = recursive_parse(query_string[1:-1], index)
    elif re.fullmatch('(.+)\s(AND|OR|AND_NOT)\s(.+)', query_string):
        parsed_args = get_op_parse(query_string)
        result = handle_operation(recursive_parse(parsed_args[0], index), recursive_parse(parsed_args[2], index), parsed_args[1])
    else:
        result = resolve_single_term(query_string, index)
    return result

        
def retreive_results(query):
    #wrapper for recursive descent
    #takes the query string returns a list of doc IDs 
    inv_ind = get_vinhs_garbage_pickle_thing()
    return recursive_parse(query, inv_ind)
