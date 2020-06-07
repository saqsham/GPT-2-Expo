# Introduction to gpt-2

Main paper released in 2019 by OpenAI ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

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


### Working

This repo contains a `keyword_encode.py` script which attempts to extract the keywords in an unsupervised manner (although you can provide your own keywords if you have them). The methodology is as follows for each text document:

1. Extract the keywords from each document as "keywords" using spaCy, which both tokenizes keywords and tags their parts-of-speech.
	* Only nouns, verbs, adjectives, and adverbs are extracted. Nouns use the raw version of the word (for best user experience when they input them manually) while the other POS use the lemmatized versions (to reduce overfitting but still provide information).
	* Proper nouns, named entities, and compound nouns count as their own keyword.
	* Pronouns and stop words are excluded from keywords.
	* Keywords are deduped.
2. Prepare the keywords in such a way that the document text is generated conditionally on the keywords.
	* Normalize the keywords (replace spaces/punctuation w/ dashes). The keywords are *not* case-normalized for best user experience when specifying keywords.
	* Shuffle the order of the keywords to prevent GPT-2 from cheating and learning when the order of the keywords should be written in the document proper.
	* For each set of processed keywords in a document, create `repeat` random combinations (default: 3) of the keywords. This serves as a data augmentation of sorts, and prevents the model from overfitting on a given set of keywords.
	* For each combination above, select a random number of *up to* `max_keywords` (default: 3), which are then shuffled, to prevent the neural network from a) learning the number of keywords as a hint to the length of the text and b) the order of the keywords in the resulting text.
3. Write the keywords, then the document for each generated set of keywords.
	* The documents are processed in batches with ray; after each batch is encoded, the batch is shuffled before writing to reduce leakage.

The default case (passing a CSV of `titles`) generates `keywords`, and outputs a `.txt` of keywords and titles.

The `keyword_decode.py` script contains functions for decoding bulk-generated encoded texts (e.g. generated through gpt-2-simple, albeit the native truncation is recommended in that use case). `decode_texts()` will extract the text from each of the specified taxonomic sections for the provided list of texts, and `decode_file()` can extract and decode all texts and write to a file.

The encoding is tokenized using [spaCy](https://spacy.io) for more robust keyword tokenization and parallelized using [ray](https://github.com/ray-project/ray) in order to massively speed up encoding on large datasets.

### Usage

- In this repo there are two datasets, first finetune it using the `train_ask.py ` then only generate text.
- run ```python train_ask.py```, it will download the model specified, and fintune your dataset over the model (124M is the deafault/hardcoded model), parameters:
    * restore_from: Set to fresh to start training from the base GPT-2, or set to latest to restart training from an existing checkpoint.
    * sample_every: Number of steps to print example output
    print_every: Number of steps to print training progress.
    * learning_rate: Learning rate for the training. (default 1e-4, can lower to 1e-5 if you have <1MB input data)
    * run_name: subfolder within checkpoint to save the model. This is useful if you want to work with multiple models (will also need to specify run_name when loading the model)
    * overwrite: Set to True if you want to continue finetuning an existing model (w/ restore_from='latest') without creating duplicate copies.
- run ```python train_encode_text.py```, it will generate encoded data in bulk (can be manipulated in the file itself), one sample is in [data](./reddit_titles_encoded.txt) directory.
- run ```python train_decode_text.py```, it will generate decoded text from encoded text, one sample is in [data](./data/reddit_titles_decoded.txt) directory.
- run ```python generate_title.py```, it will generate text from the trained model, again file can be manipulated as per the requirement with different parameters:
    * length: Number of tokens to generate (default 1023, the maximum)
    * temperature: The higher the temperature, the crazier the text (default 0.7, recommended to keep between 0.7 and 1.0)
    * top_k: Limits the generated guesses to the top k guesses (default 0 which disables the behavior; if the generated output is super crazy, you may want to set top_k=40)
    * top_p: Nucleus sampling: limits the generated guesses to a cumulative probability. (gets good results on a dataset with top_p=0.9)
    * truncate: Truncates the input text until a given sequence, excluding that sequence (e.g. if truncate='<|endoftext|>', the returned text will include everything before the first <|endoftext|>). It may be useful to combine this with a smaller length if the input texts are short.
    * include_prefix: If using truncate and include_prefix=False, the specified prefix will not be included in the returned text.
- Premade google colab notebooks are also available [10], [11] &[12].
- Check the [report](./final_report.pdf) for implementation with images.


### Other useful information

- The implementaion in this repo is done using gpt-2-simple, one can install it simply by pip
```
pip3 install gpt-2-simple
```
- You will also need to install the corresponding TensorFlow for your system (e.g. tensorflow or tensorflow-gpu). TensorFlow 2.0 is currently not supported and the package will throw an assertion if loaded, so TensorFlow 1.14/1.15 is recommended.
- gpt-2-simple is A simple Python package that wraps existing model fine-tuning and generation scripts for OpenAI's [GPT-2 text generation model](https://openai.com/blog/better-language-models/) (specifically the "small" 124M and "medium" 355M hyperparameter versions). Check the [repo](https://github.com/minimaxir/gpt-2-simple) for more information.
- For finetuning, it is strongly recommended to use a GPU, although you can generate using a CPU (albeit much more slowly)


### Differences Between gpt-2-simple And Other Text Generation Utilities

The method GPT-2 uses to generate text is slightly different than those like other packages like textgenrnn (specifically, generating the full text sequence purely in the GPU and decoding it later), which cannot easily be fixed without hacking the underlying model code. As a result:

* In general, GPT-2 is better at maintaining context over its entire generation length, making it good for generating conversational text. The text is also generally gramatically correct, with proper capitalization and few typoes.
* The original GPT-2 model was trained on a *very* large variety of sources, allowing the model to incorporate idioms not seen in the input text.
* GPT-2 can only generate a maximum of 1024 tokens per request (about 3-4 paragraphs of English text).
* GPT-2 cannot stop early upon reaching a specific end token. (workaround: pass the `truncate` parameter to a `generate` function to only collect text until a specified end token. You may want to reduce `length` appropriately.)
* Higher temperatures work better (e.g. 0.7 - 1.0) to generate more interesting text, while other frameworks work better between 0.2 - 0.5.
* When finetuning GPT-2, it has no sense of the beginning or end of a document within a larger text. You'll need to use a bespoke character sequence to indicate the beginning and end of a document. Then while generating, you can specify a `prefix` targeting the beginning token sequences, and a `truncate` targeting the end token sequence. You can also set `include_prefix=False` to discard the prefix token while generating (e.g. if it's something unwanted like `<|startoftext|>`).
* If you pass a single-column `.csv` file to `finetune()`, it will automatically parse the CSV into a format ideal for training with GPT-2 (including prepending `<|startoftext|>` and suffixing `<|endoftext|>` to every text document, so the `truncate` tricks above are helpful when generating output). This is necessary to handle both quotes and newlines in each text document correctly.
* GPT-2 allows you to generate texts in parallel by setting a `batch_size` that is divisible into `nsamples`, resulting in much faster generation. Works very well with a GPU (can set `batch_size` up to 20 on Colaboratory's K80)!
* Due to GPT-2's architecture, it scales up nicely with more powerful GPUs. For the 124M model, if you want to train for longer periods of time, GCP's P100 GPU is about 3x faster than a K80/T4 for only 3x the price, making it price-comparable (the V100 is about 1.5x faster than the P100 but about 2x the price). The P100 uses 100% of the GPU even with `batch_size=1`, and about 88% of the V100 GPU.
* If you have a partially-trained GPT-2 model and want to continue finetuning it, you can set `overwrite=True` to finetune, which will continue training and remove the previous iteration of the model without creating a duplicate copy. This can be especially useful for transfer learning (e.g. heavily finetune GPT-2 on one dataset, then finetune on other dataset to get a "merging" of both datasets).
* If your input text dataset is massive (>100 MB), you may want to preencode and compress the dataset using `gpt2.encode_dataset(file_path)`. THe output is a compressed `.npz` file which will load much faster into the GPU for finetuning.
* The 774M "large" model may support finetuning because it will cause modern GPUs to go out-of-memory (you may get lucky if you use a P100 GPU on Colaboratory). However, you can still generate from the default pretrained model using `gpt2.load_gpt2(sess, model_name='774M')` and `gpt2.generate(sess, model_name='774M')`.
* The 1558M "extra large", true model, may not work out-of-the-box with the GPU included with the Colaboratory Notebook. More testing is needed to identify optimial configurations for it.


## Apps and Examples

* [gpt2-small](https://minimaxir.com/apps/gpt2-small/) — App using the default GPT-2 124M pretrained model
* [gpt2-reddit](https://minimaxir.com/apps/gpt2-reddit/) — App to generate Reddit titles based on a specified subreddit and/or keyword(s)
* [gpt2-mtg](https://minimaxir.com/apps/gpt2-mtg/) — App to generate Magic: The Gathering cards
* [ResetEra](https://www.resetera.com/threads/i-trained-an-ai-on-thousands-of-resetera-thread-conversations-and-it-created-hot-gaming-shitposts.112167/) — Generated video game forum discussions ([GitHub w/ dumps](https://github.com/minimaxir/resetera-gpt-2))
* [/r/legaladvice](https://www.reddit.com/r/legaladviceofftopic/comments/bfqf22/i_trained_a_moreadvanced_ai_on_rlegaladvice/) — Title generation ([GitHub w/ dumps](https://github.com/minimaxir/legaladvice-gpt2))
* [Hacker News](https://github.com/minimaxir/hacker-news-gpt-2) — Tens of thousands of generated Hacker News submission titles
* [Examples which use Transformer](https://transformer.huggingface.co/)


## Modifying the Model

Please refer to [6],[7],[8] and [9].


## Further

Check this out, **[Transformers](https://github.com/huggingface/transformers)**, (formerly known as ```pytorch-transformers``` and ```pytorch-pretrained-bert```) provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5, CTRL...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over thousands of pretrained models in 100+ languages and deep interoperability between PyTorch & TensorFlow 2.0.[23]


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
<br>[24]. https://openai.com/blog/better-language-models/
<br>[25]. https://github.com/minimaxir/textgenrnn


## Team Members

see [TEAMMEMBERS.md](./TEAMMEMBERS.md)

## License 

[MIT](./LICENSE)


## Disclaimer 

This repo has no afflication or relationship with OpenAI or minimaxir, all of the sources are opensource and proper links are provied.