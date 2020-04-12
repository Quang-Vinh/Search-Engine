# Module 1 - User Interface
# Purpose: Allow a user to access the search engine capabilities

# Kivy modules
import kivy
from kivy.app import App
from kivy.config import Config
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button, ButtonBehavior
from kivy.uix.checkbox import CheckBox
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# Other libraries
import pandas as pd
import re
import textwrap
from win32api import GetSystemMetrics

# Search engine modules
from boolean_retrieval import BooleanRetrievalModel
from corpus_access import get_corpus_texts
from inverted_index import InvertedIndex, load_index
from spelling_correction import SpellingCorrector
from vector_space_model import VectorSpaceModel


class LabelButton(ButtonBehavior, Label):
    pass


class SearchScreen(GridLayout):

    # Load index and setup models
    uo_index = load_index("../models/indexes/UofO_Courses_index.pkl")
    uo_spelling_corrector = SpellingCorrector(uo_index.dictionary.words_raw)
    uo_vsm_model = VectorSpaceModel(uo_index)
    uo_bool_model = BooleanRetrievalModel(uo_index)

    reuters_index = load_index("../models/indexes/reuters_index.pkl")
    reuters_spelling_corrector = SpellingCorrector(reuters_index.dictionary.words_raw)
    reuters_vsm_model = VectorSpaceModel(reuters_index)
    reuters_bool_model = BooleanRetrievalModel(reuters_index)

    indexes = {"uo_courses": uo_index, "reuters": reuters_index}
    spelling_correctors = {
        "uo_courses": uo_spelling_corrector,
        "reuters": reuters_spelling_corrector,
    }
    vsm_models = {"uo_courses": uo_vsm_model, "reuters": reuters_vsm_model}
    bool_models = {"uo_courses": uo_bool_model, "reuters": reuters_bool_model}
    corpus_selected = "uo_courses"

    layout_content = ObjectProperty(None)

    # https://stackoverflow.com/questions/26686631/how-do-you-scroll-a-gridlayout-inside-kivy-scrollview
    def __init__(self, **kwargs):
        super(SearchScreen, self).__init__(**kwargs)
        self.layout_content.bind(minimum_height=self.layout_content.setter("height"))

    def search(self):
        """
        Onclick for button search for given search query
        """

        query = self.ids["search_query_input"].text
        self.corpus_selected = (
            "uo_courses" if self.ids["uo_courses"].active else "reuters"
        )

        # Get search results
        if self.ids["vsm"].active:
            results = self.vsm_models[self.corpus_selected].search(
                query, include_similarities=True, limit=10
            )
            docIDs = [docID for docID, _ in results]
            scores = [score for _, score in results]
        elif self.ids["boolean"].active:
            docIDs = self.bool_models[self.corpus_selected].retrieve_results(query)
            scores = [1] * len(docIDs)
        else:
            return
        print(docIDs)
        search_results = get_corpus_texts(self.corpus_selected, docIDs)

        # Get suggested queries
        suggested_queries = self.spelling_correctors[self.corpus_selected].check_query(
            query
        )

        # Update search results
        search_results["score"] = scores
        self.show_search_results(search_results)

        # Update suggested queries
        self.show_suggested_queries(suggested_queries)

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
        search_results_grid.add_widget(title)
        search_results_grid.add_widget(score)

        for docID, doc in search_results.iterrows():
            # Add course info
            body = doc["body"]
            excerpt = body[:100] if len(body) > 100 else body
            search_result = LabelButton(
                text=f"{docID} \n {excerpt}", size_hint=(1, None)
            )
            search_result.bind(on_press=self.show_search_result_popup)
            search_results_grid.add_widget(search_result)

            # Add score
            score = doc["score"]
            score_label = Label(text=str(score))
            search_results_grid.add_widget(score_label)

        return

    def show_search_result_popup(self, instance) -> None:
        """
        Open a popup for clicked on search result to show entire search result body
        """

        docID = instance.text.split("\n")[0].strip()
        if self.corpus_selected == "reuters":
            docID = int(docID)

        doc = get_corpus_texts(self.corpus_selected, [docID]).iloc[0]
        title = doc["title"]
        body = doc["body"]

        label = Label(text=textwrap.fill(body, 50))
        search_result_popup = Popup(
            title=f"{docID} - {title}",
            content=label,
            size_hint=(None, None),
            size=(400, 400),
        )
        search_result_popup.open()

        return

    def show_suggested_queries(self, suggested_queries: list) -> None:
        """
        Displays suggested queries in suggested queries grid
        """
        suggested_queries_grid = self.ids["suggested_queries_grid"]

        # Clear previous suggested queries
        suggested_queries_grid.clear_widgets()

        # Add for each each suggestion
        for i in range(0, len(suggested_queries)):
            label = LabelButton(text=f"{i+1}) {suggested_queries[i]}")
            label.bind(on_press=self.suggested_query_search)
            suggested_queries_grid.add_widget(label)

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


class SearchEngineApp(App):
    def build(self):
        return SearchScreen()


if __name__ == "__main__":

    # Set window maximized
    Config.set("graphics", "width", GetSystemMetrics(0))
    Config.set("graphics", "height", GetSystemMetrics(1) - 100)

    SearchEngineApp().run()
