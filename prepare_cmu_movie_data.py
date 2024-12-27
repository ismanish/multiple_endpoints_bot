import json
import csv
from collections import defaultdict
from datetime import datetime

def read_plot_summaries(filename):
    summaries = {}
    with open(filename, 'r') as f:
        for line in f:
            movie_id, script = line.split('\t')
            summaries[movie_id] = script
    return summaries


def read_movies_actors(filename):
    movies_actors = defaultdict(set)
    with open(filename, 'r') as f:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            if len(row) >=9:
                movie_id = row[0]
                actor_id = row[8]
                if actor_id:
                    movies_actors[movie_id].add(actor_id)
    return movies_actors


def movie_metadata_read(filename):
    movie_metadata = {}
    with open(filename, 'r') as f:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            if len(row) >=8:
                movie_id = row[0]
                title = row[2]
                year = row[3] if row[3] else None
                genres = json.loads(row[8]) if row[8] else None
                genres = list(genres.values()) if genres else None
                movie_metadata[movie_id] = {'title': title, 'year': year, 'genres': genres}
    return movie_metadata


def consolidate_data(plot_summaries, movies_actors, movie_metadata, output_file):
        plot_summaries = read_plot_summaries(plot_summaries)
        movies_actors = read_movies_actors(movies_actors)
        movie_metadata = movie_metadata_read(movie_metadata)
        consolidated_data = []
        for movie_id, summary in plot_summaries.items(): # iterate over the movie ids and summaries in plot_summaries::
            if movie_id in movies_actors and movie_id in movie_metadata:
                if movie_metadata[movie_id]['year']:        
                    year = movie_metadata[movie_id]['year'][:4]
                    if int(year) >= 2011 and len(list(movies_actors[movie_id])) >= 2:
                        movie_data = {
                            'movie_id': movie_id,   
                            'title': movie_metadata[movie_id]['title'],
                            'year': movie_metadata[movie_id]['year'],
                            'genres': movie_metadata[movie_id]['genres'],
                            'plot_summary': summary,
                            'actors': list(movies_actors[movie_id])

                        }
                        consolidated_data.append(movie_data)
        with open(output_file, 'w') as f:
            json.dump(consolidated_data, f, indent=2)

        print(f"Consolidated data saved to {output_file}\n with {len(consolidated_data)} movies")

if __name__ == '__main__':
    plot_file = "MovieSummaries/plot_summaries.txt"
    character_file = "MovieSummaries/character.metadata.tsv"
    movie_metadata_file = "MovieSummaries/movie.metadata.tsv"
    output_file = "movie_data.json"

    consolidate_data(plot_file, character_file, movie_metadata_file, output_file)