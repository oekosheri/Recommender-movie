# Recommender-movie
A movie recommender on [movie-lens](https://grouplens.org/datasets/movielens/20m/) data base served with a Flask-restful app  
This recommendation system uses the concepts of both memory and model based filtering to make top N recommendations.
The [Jupyter Notebook](https://nbviewer.jupyter.org/github/oekosheri/Recommender-movie/blob/master/Recom_walk_through.ipynb) shows a step by step 
walkthrough. The steps in the notebook have been rearranged in the three scripts: [Build_model.py](./Build_model.py), [model.py](./model.py) and 
[app.py](./app.py) to facilitate serving the recommender in service.

You can make different types of requests which will take you to different endpoints.
In the directory of Recommender-movie, run the API locally: ``` > python app.py ```.  
Using curl in the terminal for example, you can make a GET request at the URL of the API to get top n movies, similar to a particular movie
via content, collaborative or hybrid filtering:
```
 curl -X GET http://127.0.0.1:8000/movies/[basis] -d movie=[movie] -d limit=n
```
where [basis] can be content, collaborative or hybrid, [movie] is the name of a movie in the data base (it is usually followed by the year of production), 
n is an integer. For more information see the [article](https://blog.codecentric.de/en/2019/06/recommender-system-movie-lens-dataset/).

 _ex:_

```
> curl -X GET http://127.0.0.1:8000/movies/collaborative -d movie="Inception (2010)" -d limit=10
```
Then you will get a list:

```
{
    "collaborative": [
        "Shutter Island (2010)",
        "Inglourious Basterds (2009)",
        "Avatar (2009)",
        "Social Network, The (2010)",
        "District 9 (2009)",
        "Black Swan (2010)",
        "Sherlock Holmes (2009)",
        "Kick-Ass (2010)",
        "King's Speech, The (2010)",
        "Up (2009)"
    ]
}
```
Another form of recommendation can be made for a particular user with an id: [userId] to get a list of n movies, that this user has not yet rated,
but will likely rate highly. The curl request below will recommend a top 10 list to user=77:
```
> curl -X GET http://127.0.0.1:8000/users/77  -d limit=10
```
the result:
```
[
    "Cosmos (1980)",
    "Louis C.K.: Hilarious (2010)",
    "Louis C.K.: Shameless (2007)",
    "From the Earth to the Moon (1998)",
    "Harakiri (Seppuku) (1962)",
    "Godfather, The (1972)",
    "Seven Samurai (Shichinin no samurai) (1954)",
    "Stalker (1979)",
    "Frozen Planet (2011)",
    "Persona (1966)"
]
```




