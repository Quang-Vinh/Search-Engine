# Search-Engine
Search Engine project for CSI4107 - Information Retrieval. Search engine built on top of UofO courses collection and the reuters text collection.

## Quick Start

### Install dependencies
 Required packages are:  
 
 - Python 3.7
 - Kivy
 - Nltk
 - Numpy
 - Pandas
 - BeautifulSoup  

Note: Python 3.8 not supported as Kivy does not support that version yet
```
pip install -r requirements.txt
```

### Preprocessing data and setting up indexes
Next setup dictionaries and indexes by running command below which should take around 3 minutes. Additionally you can add the --knn flag to run knn algorithm on reuters to predict topics. This will take around 8 minutes total.  All preprocessed data and models are already included within the repo in collections and models. 
```
python make_data.py
python make_data.py --knn #For knn on reuters
```

### Start app
```
python search_engine_app.py
```









