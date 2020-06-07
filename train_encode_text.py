from keyword_encode import encode_keywords
import ray

ray.init(object_store_memory=100 * 1000000,
         redis_max_memory=100 * 1000000)

# generate keyword in bulk, default limit is 48000
encode_keywords(csv_path='reddit_titles.csv',
                out_path='data/reddit_titles_encoded.txt',
                category_field='subreddit',
                title_field='title',
                keyword_gen='title')

