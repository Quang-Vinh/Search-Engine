# Module 1 - User Interface
# Purpose: Allow a user to access the search engine capabilities

# Module 4 - Relevance Feedback
# Purpose: Capture documents that user finds relevant or non relevant for particular query

# Kivy modules
import kivy
from kivy.app import App
from kivy.config import Config
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button, ButtonBehavior
from kivy.uix.checkbox import CheckBox
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# Other libraries
from collections import defaultdict
from functools import partial
import pandas as pd
import pickle
import re
import textwrap
from win32api import GetSystemMetrics

# Search engine modules
from bigram_language_model import BigramLanguageModel
from boolean_retrieval import BooleanRetrievalModel
from corpus_access import contains_topic, corpora, get_corpus_texts, reuters_topics
from inverted_index import InvertedIndex
from query_completion import QueryCompleter
from spelling_correction import SpellingCorrector
from vector_space_model import VectorSpaceModel
from query_expansion import expand_query

# Other
from pathlib import Path


# Relevance feedback stored as a dictionary in form of {'query': (list of non relevant docIDs, list of relevant docIDs)} per corpus
relevance_feedback = {
    "uo_courses": defaultdict(lambda: (set(), set())),
    "reuters": defaultdict(lambda: (set(), set())),
}


class LabelButton(ButtonBehavior, Label):
    pass


# https://github.com/kivy/kivy/wiki/Scrollable-Label
class ScrollableLabel(ScrollView):
    text = StringProperty("")


class SearchScreen(GridLayout):

    # Load index and setup models
    uo_index_path = Path(__file__).parent / "../models/indexes/UofO_Courses_index.pkl"
    uo_index = pickle.load(uo_index_path.open("rb"))
    uo_spelling_corrector = SpellingCorrector(uo_index.dictionary.words_raw)
    uo_vsm_model = VectorSpaceModel(uo_index)
    uo_bool_model = BooleanRetrievalModel(uo_index)
    uo_bigram_path = (
        Path(__file__).parent / "../models/bigram_language_models/UofO_bigram_model.pkl"
    )
    uo_bigram_model = pickle.load(uo_bigram_path.open("rb"))
    uo_query_completer = QueryCompleter(uo_bigram_model)

    reuters_index_path = Path(__file__).parent / "../models/indexes/reuters_index.pkl"
    reuters_index = pickle.load(reuters_index_path.open("rb"))
    reuters_spelling_corrector = SpellingCorrector(reuters_index.dictionary.words_raw)
    reuters_vsm_model = VectorSpaceModel(reuters_index)
    reuters_bool_model = BooleanRetrievalModel(reuters_index)
    reuters_bigram_path = (
        Path(__file__).parent
        / "../models/bigram_language_models/reuters_bigram_model.pkl"
    )
    reuters_bigram_model = pickle.load(reuters_bigram_path.open("rb"))
    reuters_query_completer = QueryCompleter(reuters_bigram_model)

    indexes = {"uo_courses": uo_index, "reuters": reuters_index}
    spelling_correctors = {
        "uo_courses": uo_spelling_corrector,
        "reuters": reuters_spelling_corrector,
    }
    vsm_models = {"uo_courses": uo_vsm_model, "reuters": reuters_vsm_model}
    bool_models = {"uo_courses": uo_bool_model, "reuters": reuters_bool_model}
    query_completers = {
        "uo_courses": uo_query_completer,
        "reuters": reuters_query_completer,
    }

    # Flags for options when searching
    model_selected = "vsm"
    corpus_selected = "uo_courses"


    # https://stackoverflow.com/questions/26686631/how-do-you-scroll-a-gridlayout-inside-kivy-scrollview
    def __init__(self, **kwargs):
        super(SearchScreen, self).__init__(**kwargs)
        self.layout_content.bind(minimum_height=self.layout_content.setter("height"))

        # Setup dropdown https://kivy.org/doc/stable/api-kivy.uix.dropdown.html
        self.dropdown = DropDown()
        topics = ['Select topic'] + list(reuters_topics)
        for topic in topics:
            btn = Button(text=topic, size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
            self.dropdown.add_widget(btn)
        btn_dropdown = self.ids['btn_dropdown']
        btn_dropdown.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=lambda isinstance, x: setattr(btn_dropdown, 'text', x))

        layout_content = ObjectProperty(None)

        return

    def search(self):
        """
        Onclick for button search for given search query
        """
        query = self.ids["search_query_input"].text
        
        query_expansion_popup = Popup()
        query_expansion_popup.title = "e  x  p  a  n  d"
        query_expansion_popup.title_align = "center"
        query_expansion_popup.size_hint = None, None
        query_expansion_popup.size = 200, 200
        
        self.corpus_selected = (
            "uo_courses" if self.ids["uo_courses"].active else "reuters"
        )

        # Get search results
        if self.ids["vsm"].active:
            query_expansion_popup.open()
            relevant_doc_ids = relevance_feedback[self.corpus_selected][query][1]
            non_relevant_doc_ids = relevance_feedback[self.corpus_selected][query][0]
            results = self.vsm_models[self.corpus_selected].search(
                query,
                include_similarities=True,
                limit=10,
                relevant_doc_ids=relevant_doc_ids,
                non_relevant_doc_ids=non_relevant_doc_ids,
            )
            docIDs = [docID for docID, _ in results]
            scores = [score for _, score in results]
        elif self.ids["boolean"].active:
            query_expansion_popup.open()
            docIDs = self.bool_models[self.corpus_selected].retrieve_results(query)
            scores = [1] * len(docIDs)
        else:
            return

        # Get suggested queries
        suggested_queries = self.spelling_correctors[self.corpus_selected].check_query(
            query, limit=5
        )

        # Update search results
        topic = self.ids['btn_dropdown'].text

        search_results = corpora[self.corpus_selected].loc[docIDs]
        search_results["score"] = scores

        # Get subset of texts of given topic
        if topic != 'Select topic' and self.corpus_selected == "reuters":
            topic_filter = contains_topic(search_results["topics"], topic)
            search_results = search_results.loc[topic_filter]
        
        self.show_search_results(search_results)

        # Update suggested queries
        self.show_suggested_queries(suggested_queries)

        # Add bind again, bugs for some reason
        self.ids['btn_dropdown'].bind(on_release=self.dropdown.open)

        return

    def show_search_results(self, search_results: pd.DataFrame) -> None:
        """
        Displays search results in the search result grid
        """
        search_results_grid = self.ids["search_results_grid"]

        # Clear previous search results
        search_results_grid.clear_widgets()

        # Add table titles
        title = Label(text="Search Results")
        score = Label(text="Score")
        relevance = Label(text="Relevance", size_hint_x=None)
        search_results_grid.add_widget(title)
        search_results_grid.add_widget(score)
        search_results_grid.add_widget(relevance)

        for docID, doc in search_results.iterrows():
            # Add course info
            title = doc["title"]
            title = title[:50] if len(title) > 50 else title
            body = doc["body"]
            excerpt = body[:150] if len(body) > 150 else body
            excerpt = textwrap.fill(excerpt, 75)
            search_result = LabelButton(text=f"{docID} - {title}\n{excerpt}", size_hint=(1, None))
            search_result.bind(on_press=self.show_search_result_popup)
            search_results_grid.add_widget(search_result)

            # Add score
            score = doc["score"]
            score_label = Label(text=str(score))
            search_results_grid.add_widget(score_label)

            # Add relevance buttons
            btn_grid = GridLayout(cols=2, size_hint_x=None)
            yes_btn = CheckBox(
                group=str(docID), on_press=partial(self.toggle_relevance, docID, True)
            )
            no_btn = CheckBox(
                group=str(docID),
                color=[10, 1, 1, 1],
                on_press=partial(self.toggle_relevance, docID, False),
            )
            btn_grid.add_widget(yes_btn)
            btn_grid.add_widget(no_btn)
            search_results_grid.add_widget(btn_grid)

        return

    def toggle_relevance(self, docID: str, relevant: bool, instance) -> None:
        """Updates in memory relevance feedback data structure 
        
        Arguments:
            docID {str} -- docID
            relevant {bool} -- If button was relevant or non relevant
            instance {[type]} -- Checkbox instance
        """
        query = self.ids["search_query_input"].text

        if relevant and instance.active:
            relevance_feedback[self.corpus_selected][query][1].add(docID)
            relevance_feedback[self.corpus_selected][query][0].discard(docID)
        elif not relevant and instance.active:
            relevance_feedback[self.corpus_selected][query][1].discard(docID)
            relevance_feedback[self.corpus_selected][query][0].add(docID)
        elif not instance.active:
            relevance_feedback[self.corpus_selected][query][1].discard(docID)
            relevance_feedback[self.corpus_selected][query][0].discard(docID)
        return

    def show_search_result_popup(self, instance) -> None:
        """
        Open a popup for clicked on search result to show entire search result body
        """

        docID = instance.text.split("-")[0].strip()
        if self.corpus_selected == "reuters":
            docID = int(docID)

        doc = get_corpus_texts(self.corpus_selected, [docID]).iloc[0]
        title = doc["title"]
        body = doc["body"]

        # Popup
        search_result_popup = Popup()
        search_result_popup.title = f"{docID} - {title}"
        search_result_popup.title_align = "center"
        search_result_popup.size_hint = None, None
        search_result_popup.size = 1400, 800

        label = Label(text=textwrap.fill(body, 170))

        # label_scroll = ScrollableLabel()
        # label_scroll.text = body
        # label_scroll.size_hint_y = None
        # label_scroll.height = 500
        # label_scroll.text_size = 200, None
        # search_result_popup = Popup(
        #     title=f"{docID} - {title}",
        #     content=label_scroll,
        #     size_hint=(None, None),
        #     size=(600, 600)
        # )
        search_result_popup.add_widget(label)
        search_result_popup.open()

        return

    def show_suggested_queries(self, suggested_queries: list) -> None:
        """
        Displays suggested queries in suggested queries grid
        """
        suggested_queries_grid = self.ids["suggested_queries_grid"]

        # Clear previous suggested queries
        suggested_queries_grid.clear_widgets()

        # Add each suggestion
        for i in range(0, len(suggested_queries)):
            label = LabelButton(text=f"{i+1}) {suggested_queries[i]}")
            label.bind(on_press=self.suggested_query_search)
            suggested_queries_grid.add_widget(label)
        return

    def suggested_query_search(self, instance) -> None:
        """
        Make new search with suggested query
        """
        # Set new query
        suggested_query = re.sub(r"[0-9]*\) ", "", instance.text, 1)
        self.ids["search_query_input"].text = suggested_query

        # Search
        self.search()
        return

    def show_query_completions(self, query: str) -> None:
        """Predict completed queries for given query and output them to the UI 
        
        Arguments:
            query {str} -- Query to predict completed queries from
        """
        query_completions_grid = self.ids["query_completions_grid"]

        # Get completed queries given corpus
        completed_queries = self.query_completers[self.corpus_selected].complete_query(
            query
        )

        # Clear previous completed queries
        query_completions_grid.clear_widgets()

        # Add each completed query
        for (i, query) in enumerate(completed_queries):
            label = LabelButton(text=f"{i+1}) {query}")
            label.bind(on_press=self.complete_query_search)
            query_completions_grid.add_widget(label)
        return

    def complete_query_search(self, instance) -> None:
        """Sets current query in search text input to new query 
        
        Arguments:
            query {str} -- New query
        """
        completed_query = re.sub(r"[0-9]*\) ", "", instance.text, 1)
        self.ids["search_query_input"].text = completed_query
        self.search()
        return

    def toggle_search_model(self) -> None:
        """Toggles selected search model
        """
        self.model_selected = "vsm" if self.ids["vsm"].active else "boolean"
        return

    def toggle_corpus(self) -> None:
        """Toggles selected corpus
        """
        self.corpus_selected = (
            "uo_courses" if self.ids["uo_courses"].active else "reuters"
        )
        return


class SearchEngineApp(App):
    def build(self):
        return SearchScreen()


if __name__ == "__main__":
    # Set window maximized
    Config.set("graphics", "window_state", "maximized")

    SearchEngineApp().run()
