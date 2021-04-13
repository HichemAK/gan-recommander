import pandas as pd

genres = ["unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western",]

head = ["movieId", "title", "date", "empty", "link"]
csv_movies = pd.read_csv("u.item", sep='|', names=head+genres)

csv_movies.drop("empty", 1, inplace=True)
csv_movies.drop("link", 1, inplace=True)
csv_movies.drop("date", 1, inplace=True)

for genre in genres:
    csv_movies[genre] = csv_movies[genre].apply(lambda x : genre if x else '')

csv_movies["genre"] = csv_movies[genres].apply(lambda x : '|'.join([y for y in x if len(y) > 0]), axis=1)
csv_movies.drop(genres, axis=1, inplace=True)

print(csv_movies.head())

csv_movies.to_csv("movies.csv", index=False)