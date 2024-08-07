import os
import obonet
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict
import sys
import pickle
import numpy as np
import shutil

# default paths
file_dir = os.path.dirname(os.path.abspath(__file__))
default_go_obo = os.path.join(file_dir, "go-basic.obo")
default_obo_pkl = os.path.join(file_dir, "obo.pkl")

class ObOTools:

    def __init__(self,
        go_obo: str = default_go_obo, # Path to the go.obo file
        obo_url: str = "http://purl.obolibrary.org/obo/go/go-basic.obo", # URL to the go.obo file
        obo_pkl: str = default_obo_pkl, # Path to the obo.pkl file
        add_part_of:bool = True, # whether to add the part_of relationship to the is_a.csv file
        verbose:bool = True # whether to print the warning message
        ):
        self.go_obo = go_obo
        self.obo_url = obo_url
        self.obo_pkl = obo_pkl
        self.add_part_of = add_part_of
        self.verbose = verbose
        self.aspects = ['BPO', 'CCO', 'MFO']
        self.is_a_dict, self.alt_id_dict, self.term2aspect, self.obo_net \
            = self.init_obo(go_obo=self.go_obo, obo_url=self.obo_url)

    def init_obo(self, go_obo:str=None, obo_url:str=None):
        """
        Initialize the obo file
        """
        # check if the obo.pkl file exists
        if os.path.exists(self.obo_pkl):
            if self.verbose:
                print(f"Loading the obo.pkl file from {self.obo_pkl}")
            is_a_dict, alt_id_dict, term2aspect, obo_net = self.load_obo_pkl()
            obo_version = obo_net.graph['data-version']
            if self.verbose:
                print(f"The obo file version is {obo_version}")
            return is_a_dict, alt_id_dict, term2aspect, obo_net
        
        # load from the obo file
        obo_path = go_obo
        if obo_path is None or not os.path.exists(obo_path):
            obo_path = obo_url
            
        if self.verbose:
            print(f"Loading the obo file from {obo_path}")

        obo = obonet.read_obo(obo_path)
        alt_id_dict = self.creat_alt_id_dict(obo)
        term2aspect = self.create_ns_dict(obo)
        is_a_dict = self.create_is_a_dict(obo, term2aspect, alt_id_dict, self.add_part_of)
        obo_net = obo
        # save the obo.pkl file
        if self.verbose:
            print(f"Saving the obo.pkl file to {self.obo_pkl} for future use")
            obo_version = obo_net.graph['data-version']
            print(f"THe obo file version {obo_version}")
        obo_pkl = {"is_a_dict":is_a_dict, "alt_id_dict":alt_id_dict, "term2aspect":term2aspect, "obo_net":obo_net}
        pickle.dump(obo_pkl, open(self.obo_pkl, "wb"))
        return is_a_dict, alt_id_dict, term2aspect, obo_net
            
    def update_obo(self, go_obo:str=None, obo_url:str=None, newest:bool=False):

        if go_obo is not None and os.path.exists(go_obo):
            # remove the old go_obo file
            if os.path.exists(go_obo) and go_obo != self.go_obo:
                shutil.copyfile(go_obo, self.go_obo)
            if os.path.exists(self.obo_pkl):
                os.remove(self.obo_pkl)
            self.is_a_dict, self.alt_id_dict, self.term2aspect, self.obo_net \
                = self.init_obo(go_obo=self.go_obo, obo_url=None)
            return
        
        if newest:
            obo_url = "http://purl.obolibrary.org/obo/go/go-basic.obo"

        if obo_url is not None:
            if os.path.exists(self.obo_pkl):
                os.remove(self.obo_pkl)
            obo_path = obo_url
            self.is_a_dict, self.alt_id_dict, self.term2aspect, self.obo_net \
                = self.init_obo(go_obo=None, obo_url=obo_path)
            return            

    def load_obo_pkl(self):
        """
        Load the obo.pkl file
        """
        obo_pkl = pickle.load(open(self.obo_pkl, "rb"))
        is_a_dict = obo_pkl["is_a_dict"]
        alt_id_dict = obo_pkl["alt_id_dict"]
        term2aspect = obo_pkl["term2aspect"]
        obo_net = obo_pkl["obo_net"]
        return is_a_dict, alt_id_dict, term2aspect, obo_net
    
    def goID2name(self, go_id:str):
        """
        Get the go name from the go id

        Args:
            go_id: the go id
        
        Returns:
            go_name: the go name
        """
        if go_id not in self.obo_net.nodes:
            return 'nan'
        go_name = self.obo_net.nodes[go_id]['name']
        return go_name

    def creat_alt_id_dict(self, obo):
        alt_id_dict = dict()
        for term_id, data in obo.nodes(data=True):
            if term_id not in alt_id_dict:
                alt_id_dict[term_id] = term_id
            if 'alt_id' in data:
                for alt_id in data['alt_id']:
                    alt_id_dict[alt_id] = term_id
        return alt_id_dict
    
    def create_ns_dict(self, obo):
        term2aspect = dict()
        aspect_map = {'biological_process': 'BPO', 'molecular_function': 'MFO', 'cellular_component': 'CCO'}
        for term_id, data in obo.nodes(data=True):
            if 'namespace' in data:
                term2aspect[term_id] = aspect_map[data['namespace']]
        return term2aspect

    def find_all_parents(self, term_id:str, is_a_dict:dict, cache=dict())->dict:
        parents = set()
        if term_id in cache:
            return cache[term_id]
        if term_id in is_a_dict:
            parents.update(is_a_dict[term_id])
            for parent in is_a_dict[term_id]:
                parents.update(self.find_all_parents(parent, is_a_dict))
        cache[term_id] = parents
        return parents

    def create_is_a_dict(self, obo, aspect_dict:dict, alt_id_dict:dict=None, add_part_of:bool=True)->dict:
        is_a_dict = dict()
        for term_id, data in obo.nodes(data=True):

            # replace term_id with its main id if it is an alt_id
            if alt_id_dict is not None:
                if term_id in alt_id_dict:
                    term_id = alt_id_dict[term_id]
            term_aspect = aspect_dict[term_id]

            if term_id not in is_a_dict:
                is_a_dict[term_id] = set()

            # add is_a parent
            if 'is_a' in data:
                for parent_term_id in data['is_a']:
                    parent_term_aspect = aspect_dict[parent_term_id]
                    if term_aspect == parent_term_aspect:
                        is_a_dict[term_id].add(parent_term_id)

            # add part_of parent
            if add_part_of:
                if 'relationship' in data:
                    for record in data['relationship']:
                        rel, parent_term_id = record.split(' ')
                        if rel == 'part_of':
                            parent_term_aspect = aspect_dict[parent_term_id]
                            if term_aspect == parent_term_aspect:
                                is_a_dict[term_id].add(parent_term_id)

        # create a dict of all ancestors
        is_a_dict_all = dict()
        for term_id in is_a_dict:
            is_a_dict_all[term_id] = self.find_all_parents(term_id, is_a_dict)
        return is_a_dict_all
        
    def update_parent(self, protein_name:str, cur_terms:set)->Tuple[str,set]:
        """
        Update the cur_terms to the parents of the cur_terms

        Args:
            protein_name: the name of the protein
            cur_terms: the current terms of the protein
        
        Returns:
            protein_name: the name of the protein
            final_terms: the updated terms of the protein
        """
        final_terms = set()
        for term in cur_terms:
            if term not in self.alt_id_dict:
                # warning: the term is not in the alt_id_dict, the obo file may be outdated
                if self.verbose:
                    print(f"Warning: {term} is not in the alt_id_dict, the obo file may be outdated, please check protein: {protein_name}")
                final_terms.add(term)
            else:
                all_parent = self.is_a_dict[self.alt_id_dict[term]].copy()
                all_parent.add(self.alt_id_dict[term])
                final_terms.update(all_parent)
        return protein_name, final_terms
    
    def backprop_terms(self, cur_terms:set)->set:
        if isinstance(cur_terms, List):
            cur_terms = set(cur_terms)
        final_terms = set()
        for term in cur_terms:
            if term not in self.alt_id_dict:
                # warning: the term is not in the alt_id_dict, the obo file may be outdated
                if self.verbose:
                    print(f"Warning: skipping {term} because it was not in the alt_id_dict", file=sys.stderr)
                final_terms.add(term)
            else:
                all_parent = self.is_a_dict[self.alt_id_dict[term]].copy()
                all_parent.add(self.alt_id_dict[term])
                final_terms.update(all_parent)
        return final_terms
    
    def backprop_cscore(self, go_cscore:dict, min_cscore:float=None, sorting=True)->Dict[str, float]:
        """
        backproprate the child cscore to the parents
        parent cscore should not be smaller than the child cscore
        if the parent cscore is smaller than the child cscore, update the child cscore to the parent cscore

        Args:
            go_cscore: the go_cscore dict where the key is the go term and the value is the cscore
            min_cscore: the minimum cscore, if the cscore is smaller than the min_cscore, the cscore will be set to 0
        return:
            go_cscore: the updated go_cscore dict
        """
        if min_cscore is not None:
            filter_score = {term:cscore for term, cscore in go_cscore.items() if cscore > min_cscore}
            backprop_cscore = filter_score.copy()
        else:
            filter_score = go_cscore
            backprop_cscore = go_cscore.copy()

        for term in filter_score:
            if term not in self.alt_id_dict:
                # warning: the term is not in the alt_id_dict, the obo file may be outdated
                if self.verbose:
                    print(f"Warning: skipping {term} because it was not in the alt_id_dict, the obo file may be outdated.", file=sys.stderr)
                continue
            # replace the alt_id with the main_id, and get all the parents for the term
            all_parent = self.is_a_dict[self.alt_id_dict[term]].copy()
            # loop through all the parents, if the parent is not in the backprop_cscore or the parent cscore is smaller than the child cscore
            # update the child cscore to the parent cscore
            for parent in all_parent:
                if parent not in backprop_cscore or backprop_cscore[parent] < backprop_cscore[term]:
                    backprop_cscore[parent] = backprop_cscore[term]
        
        if sorting:
            # high to low
            backprop_cscore = {k: v for k, v in sorted(backprop_cscore.items(), key=lambda item: item[1], reverse=True)}

        return backprop_cscore
    
    def get_aspect_terms(self, aspect:str):
        """
        Get all the go terms for the aspect

        Args:
            aspect: the aspect, BP, MF, CC
        
        Returns:
            go_terms: a set of go terms
        """
        go_terms = set()
        for term in self.term2aspect:
            if self.term2aspect[term] == aspect:
                go_terms.add(term)
        return go_terms
    
    def get_aspect(self,term:str):
        """
        Get the aspect of the term

        Args:
            term: the go term
        
        Returns:
            aspect: the aspect of the term
        """
        aspect =  self.term2aspect.get(term, None)
        return aspect
    
    def generate_child_matrix(self, term_list:List[str]):
        """
        Generate the child matrix for the aspect

        Args:
            go2vec: the go2vec dict where the key is the go term and the value is the index of the go term in the embedding matrix
        
        Returns:
            child_matrix: the child matrix for the aspect where child_matrix[i][j] = 1 if the jth GO term is a subclass of the ith GO term else 0
        """
        
        training_terms = term_list
        #CM_ij = 1 if the jth GO term is a subclass of the ith GO term
        child_matrix = np.zeros((len(training_terms), len(training_terms)))
        # fill diagonal with 1
        np.fill_diagonal(child_matrix, 1)
        for i, term in enumerate(training_terms):
            for j, child in enumerate(training_terms):
                if i == j:
                    continue
                if term in self.is_a_dict[child]:
                    child_matrix[i][j] = 1
        return child_matrix
    
    def toplogical_child_matrix(self, term_list:List[str]):
        """
        Generate the child matrix for the aspect
        the input term_list should be topologically sorted
        Then we can ignore these i rows on the bottom of the matrix where the sum of the row is 1 (itself)

        Args:
            term_list (List[str]): selected go terms for the aspect
        return:
            child_matrix: the child matrix for the aspect where child_matrix[i][j] = 1 if the jth GO term is a subclass of the ith GO term else 0
        """

        # check if the term_list is topologically sorted
        # if not, raise the error
        sorted_terms,leafs = self.top_sort(term_list, return_leaf=True)
        assert len(sorted_terms) == len(term_list), "The input term_list is not topologically sorted"
        for i, term in enumerate(sorted_terms):
            assert term == term_list[i], f"The {i} idx term:{term_list[i]} is not the same as the sorted term:{term}"

        training_terms = term_list
        #CM_ij = 1 if the jth GO term is a subclass of the ith GO term
        child_matrix = np.zeros((len(training_terms), len(training_terms)))
        # fill diagonal with 1
        np.fill_diagonal(child_matrix, 1)
        for i, term in enumerate(training_terms):
            for j, child in enumerate(training_terms):
                if i == j:
                    continue
                if term in self.is_a_dict[child]:
                    child_matrix[i][j] = 1
        num_leaf = len(leafs)
        child_matrix = child_matrix[:-num_leaf,:]
        return child_matrix
    
    def get_num_leaf(self, term_list:List[str]):
        """
        Get the number of leaf go terms for the aspect

        Args:
            term_list (List[str]): selected go terms for the aspect
        
        Returns:
            num_leaf (int): the number of leaf go terms for the aspect
        """
        term_list = sorted(term_list)
        set_term_list = set(term_list)

        # reverse the is_a_dict to get the child dict
        child_dict = defaultdict(set)
        for child, parent_set in self.is_a_dict.items():
            for parent in parent_set:
                if parent in set_term_list and child in set_term_list:
                    child_dict[child].add(parent)

        # create a dict to store the indegree of each go term
        indegree = {term:0 for term in term_list}
        # sort the indegree dict by the key
        indegree = dict(sorted(indegree.items(), key=lambda item: item[0]))
        
        # loop through the child dict to get the indegree of each go term
        for parent, child_set in child_dict.items():
            for child in child_set:
                indegree[child] += 1

        queue = [ term for term in term_list if indegree[term] == 0 ]

        return len(queue)
    
    def top_sort(self, term_list:List[str], return_leaf:bool=False)->List[str]:
        """
        Topological sort the go terms

        Args:
            term_list (List[str]): selected go terms for the aspect

        Returns:
            sorted_terms (List[str]): sorted go terms
        """
        term_list = sorted(term_list)
        set_term_list = set(term_list)

        # reverse the is_a_dict to get the child dict
        child_dict = defaultdict(set)
        for child, parent_set in self.is_a_dict.items():
            for parent in parent_set:
                if parent in set_term_list and child in set_term_list:
                    child_dict[child].add(parent)

        # create a dict to store the indegree of each go term
        indegree = {term:0 for term in term_list}
        # sort the indegree dict by the key
        indegree = dict(sorted(indegree.items(), key=lambda item: item[0]))
        
        # loop through the child dict to get the indegree of each go term
        for parent, child_set in child_dict.items():
            for child in child_set:
                indegree[child] += 1
        
        # find the go terms with indegree 0 (leaves) and add them to the queue
        indexes = []
        visited = set()

        queue = [ term for term in term_list if indegree[term] == 0 ]
        leafs = queue.copy()

        # for each element of the queue increment visits, add them to the list of ordered nodes
        # and decrease the in-degree of the neighbor nodes
        # and add them to the queue if they reach in-degree == 0
        while queue:
            node = queue.pop(0)
            visited.add(node)
            indexes.append(node)
            parents = self.is_a_dict[node].copy()
            parents = parents.intersection(set_term_list)
            # sort the parents
            parents = sorted(parents)
            if parents:
                for parent in parents:
                    indegree[parent] -= 1
                    if indegree[parent] == 0:
                        queue.append(parent)

        if len(visited) != len(term_list):
            print("Warning: the graph is not a DAG, the topological sort may not be correct", file=sys.stderr)
        else:
            if return_leaf:
                return indexes[::-1], leafs
            return indexes[::-1]

