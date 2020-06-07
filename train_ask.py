import gpt_2_simple as gpt2
import os
import requests

# downloads the specified model if not available
# Models:- 124M 355M 774M 1558M
model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print("Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


file_name = "reddit_titles.csv"
if not os.path.isfile(file_name):
	print("dataset not available")


sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
            dataset=file_name,
            model_name='124M',
            steps=500,
            restore_from='fresh',
            run_name='run1',
            print_every=10,
            sample_every=100
            )
