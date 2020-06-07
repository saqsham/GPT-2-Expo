# Introduction to gpt-2

Code and models from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

You can read about GPT-2 and its staged release in OpenAI's [original blog post](https://blog.openai.com/better-language-models/), [6 month follow-up post](https://openai.com/blog/gpt-2-6-month-follow-up/), and [final post](https://www.openai.com/blog/gpt-2-1-5b-release/).

OpenAI has also [released a dataset](https://github.com/openai/gpt-2-output-dataset) for researchers to study their behaviors.


## gpt-2 models

- There are currently 4 models which were released by OpenAI, small being 124M, medium 355M, large 774M and the largest being 1558M.
- This [link](https://storage.googleapis.com/gpt-2/) contains all the models, data etc. whatever OpenAI has released relating to gpt-2, just append the string between <Key></Key> to the url to navigate and download the required files or go their [repo](https://github.com/openai/gpt-2) and run the download_model.py to easily download the models or use [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple) to download the model (train_ask.py in this repo)
- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination. The models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.[2]


## gpt-2 architecture and how it works

The actual Transformer architecture GPT-2 uses is very complicated to explain (here’s a [great lecture](http://www.peterbloem.nl/blog/transformers)). For the purposes of finetuning, since we can’t modify the architecture, it’s easier to think of GPT-2 as a [black box](https://en.wikipedia.org/wiki/Black_box), taking in inputs and providing outputs. Like [previous forms of text generators](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), the inputs are a sequence of tokens, and the outputs are the probability of the next token in the sequence, with these probabilities serving as weights for the AI to pick the next token in the sequence. In this case, both the input and output tokens are [byte pair encodings](https://en.wikipedia.org/wiki/Byte_pair_encoding), which instead of using character tokens (slower to train but includes case/formatting) or word tokens (faster to train but does not include case/formatting) like most RNN approaches, the inputs are “compressed” to the shortest combination of bytes including case/formatting, which serves as a compromise between both approaches but unfortunately adds randomness to the final generation length. The byte pair encodings are later decoded into readable text for human generation.[4]


## Generating dataset

- Getting titles from reddit using google [BigQuery](https://console.cloud.google.com/bigquery).[13]
- Using BigQuery is free, it's limitedly free, please check your billing through the console for more details

```
#standardSQL
WITH
  subreddits AS (
  SELECT
    subreddit
  FROM
    `fh-bigquery.reddit_posts.2019_08`
  WHERE
    score >= 5
    AND subreddit NOT IN ("me_irl",
      "2meirl4meirl",
      "anime_irl",
      "furry_irl",
      "cursedimages")
  GROUP BY
    subreddit
  ORDER BY
    APPROX_COUNT_DISTINCT(author) DESC
  LIMIT
    1000 )
    

SELECT
  subreddit,
  REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(title, '&amp;', '&'), '&lt;', '<'), '&gt;', '>'), '�', '') as title
FROM (
  SELECT
    subreddit,
    title,
    ROW_NUMBER() OVER (PARTITION BY subreddit ORDER BY score DESC) AS score_rank
  FROM
    `fh-bigquery.reddit_posts.2019_08`
  WHERE
    subreddit IN (SELECT subreddit FROM subreddits) )
WHERE
  score_rank <= 18
ORDER BY subreddit
```


## Usage and/or Simple Implementaion

- run ```train_ask.py```, it will download the model specified, and fintune your dataset over the model (124M is the deafault/hardcoded model), parameters:
    * restore_from: Set to fresh to start training from the base GPT-2, or set to latest to restart training from an existing checkpoint.
    * sample_every: Number of steps to print example output
    print_every: Number of steps to print training progress.
    * learning_rate: Learning rate for the training. (default 1e-4, can lower to 1e-5 if you have <1MB input data)
    * run_name: subfolder within checkpoint to save the model. This is useful if you want to work with multiple models (will also need to specify run_name when loading the model)
    * overwrite: Set to True if you want to continue finetuning an existing model (w/ restore_from='latest') without creating duplicate copies.
- run ```train_encode_text.py```, it will generate encoded data in bulk (can be manipulated in the file itself), one sample is in [data](./reddit_titles_encoded.txt) directory.
- run ```train_decode_text.py```, it will generate decoded text from encoded text, one sample is in [data](./data/reddit_titles_decoded.txt) directory.
- run ```generate_title.py```, it will generate text from the trained model, again file can be manipulated as per the requirement with different parameters:
    * length: Number of tokens to generate (default 1023, the maximum)
    * temperature: The higher the temperature, the crazier the text (default 0.7, recommended to keep between 0.7 and 1.0)
    * top_k: Limits the generated guesses to the top k guesses (default 0 which disables the behavior; if the generated output is super crazy, you may want to set top_k=40)
    * top_p: Nucleus sampling: limits the generated guesses to a cumulative probability. (gets good results on a dataset with top_p=0.9)
    * truncate: Truncates the input text until a given sequence, excluding that sequence (e.g. if truncate='<|endoftext|>', the returned text will include everything before the first <|endoftext|>). It may be useful to combine this with a smaller length if the input texts are short.
    * include_prefix: If using truncate and include_prefix=False, the specified prefix will not be included in the returned text.
- Premade google colab notebooks are also available [10][11][12].


## Modifying the Model

PLease refer to [6],[7],[8] and [9].


## Further

Check this out, **[Transformers]**(https://github.com/huggingface/transformers), (formerly known as ```pytorch-transformers``` and ```pytorch-pretrained-bert```) provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5, CTRL...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over thousands of pretrained models in 100+ languages and deep interoperability between PyTorch & TensorFlow 2.0.[23]


## References (bare links)

[1]. https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
<br>[2]. https://github.com/openai/gpt-2
<br>[3]. https://github.com/minimaxir/gpt-2-simple
<br>[4]. https://minimaxir.com/2019/09/howto-gpt2/
<br>[5]. http://www.peterbloem.nl/blog/transformers
<br>[6]. https://github.com/google/sentencepiece
<br>[7]. https://github.com/openai/gpt-2/issues/114
<br>[8]. https://github.com/soaxelbrooke/python-bpe
<br>[9]. https://huggingface.co/transformers/model_doc/gpt2.html
<br>[10]. https://colab.research.google.com/drive/1RugXCYDcMvSACYNt9j0kB6zzqRKzAbBn#scrollTo=sUmTooTW3osf
<br>[11]. https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce#scrollTo=H7LoMj4GA4n_
<br>[12]. https://colab.research.google.com/drive/1qxcQ2A1nNjFudAGN_mcMOnvV9sF_PkEb#scrollTo=KBkpRgBCBS2_
<br>[13]. https://console.cloud.google.com/bigquery
<br>[14]. https://www.reddit.com/r/legaladviceofftopic/comments/bxi869/i_trained_an_ai_to_generate_the_ultimate/
<br>[15]. https://minimaxir.com/apps/gpt2-reddit/
<br>[16]. https://github.com/minimaxir/reddit-gpt-2-cloud-run
<br>[17]. https://github.com/minimaxir/gpt-2-keyword-generation
<br>[18]. https://www.reddit.com/r/SubSimulatorGPT2/
<br>[19]. https://www.reddit.com/r/SubSimulatorGPT2/comments/btfhks/what_is_rsubsimulatorgpt2/
<br>[20]. https://huggingface.co/gpt2
<br>[21]. https://huggingface.co/
<br>[22]. https://huggingface.co/models
<br>[23]. https://github.com/huggingface/transformers


## Team Members

see [TEAMMEMBERS.md](./TEAMMEMBERS.md)

## License 

[MIT](./LICENSE)


## Disclaimer 

This repo has no afflication or relationship with OpenAI or minimaxir, all of the sources are opensource and proper links are provied.