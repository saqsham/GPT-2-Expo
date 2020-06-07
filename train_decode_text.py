from keyword_decode import decode_file

decode_file(file_path='data/reddit_titles_encoded.txt',
            out_file='data/reddit_titles_decoded.txt',
            sections=['title'],
            start_token="<|startoftext|>",
            end_token="<|endoftext|>")