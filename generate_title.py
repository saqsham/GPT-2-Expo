import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')

gpt2.generate(sess,
            run_name='run1',
            length=100,
            temperature=0.7,
            top_k=40,
            nsamples=10git ,
            batch_size=10,
            prefix="<|startoftext|>",
            truncate="<|endoftext|>",
            include_prefix=False,
            sample_delim=''
            )