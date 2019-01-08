
params = {}

params['RUN_ON_GPU'] = True


params['shell_print'] = True
params['run_eval'] = True

params['max_doc_len'] = 0
params['max_sent_len'] = 0
params['max_train_len'] = 0 # we update it in the main function based on doc_lens
params['max_dev_len'] = 0
params['max_test_len'] = 0

params['num_epochs'] = 100 #Default: 50
params['batch_size'] = 32  #Default: 32
params['test_batch_size'] = 1
params['dropout_rate'] = 0.5
params['relation_size'] = 50
params['lstm_size'] = 300 #300
params['emb_size'] = 50
params['seed'] = 1234
params['model'] = 'coh_lstm' # 'coh_word', 'coh_lstm' ,'lstm'
params['lr'] = 0.001 # default: 0.001
params['clip'] = 1.0
params['emb_type'] = 'kaveh' #  kaveh, glove
params['weight_decay'] = 0.0
params['utt_size'] = 3

params['voc_size'] = 4000

params['padding_level']= 'sentence' # 'sentence', 'document'
params['padding_place'] = 'post' # 'pre', 'post' 
params['remove_stopwords'] = False
params['keep_pronouns'] = False # pronouns are part of stopwords, should we remove them from the stopwords or keep them in documents?

params['result_table'] = 1
