# coding: utf-8

# ## Utility
#%%
from params import params

#%%
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if params['RUN_ON_GPU']:
    import torch.cuda as cuda


import warnings
warnings.filterwarnings('ignore')




import torch.nn.functional as F
import random
import torch
random.seed(params['seed'])
torch.manual_seed(params['seed'])


import numpy as np 
from utility import drawProgressBar

import nea.asap_reader as dataset

import codecs
import nltk
import re
num_regex = re.compile(u'^[+-]?[0-9]+\.?[0-9]*$')

def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)    
    return tokens

def is_number(token):
    return bool(num_regex.match(token))


import string

#snowball stopwords : http://snowball.tartarus.org/algorithms/english/stop.txt
# 174 stopwrods in snowball
snowball_stopwords ="i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing would should could ought i'm you're he's she's it's we're they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't let's that's who's what's here's there's when's where's why's how's a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very"
snowball_stopwords =snowball_stopwords.split()
stopwords = []
# we do tokenization on stopwords because we tokenized texts before filtering stopwords
for sw in snowball_stopwords:
    for t in tokenize(sw):
        if t not in stopwords:
            stopwords.append(t)

if params['keep_pronouns'] == True:
    
    pronouns = ['i','me','we','us','you','she','her','him','he','it','they','them','myself','ourselves',
                'yourself','yourselves','himself','herself','itself','themselves']

    stopwords = set(stopwords) - set(pronouns)           
    
    stopwords = list(stopwords)


punct = [t for t in string.punctuation]

stopwords += punct

stopwords += [u'``',u"''",u"lt",u"gt"]

def filter_stopwords(content,sents):

    filterd_content, filterd_sents = [], []
    
    for word in content:
       
        if word not in stopwords:
        
            filterd_content.append(word)
    
        
    for sent in sents:
    
        filterd_sent = []
        
        for word in sent:
        
            if word not in stopwords:
            
                filterd_sent.append(word)
        
        filterd_sents.append(filterd_sent)

    return filterd_content, filterd_sents


def load(file_path, prompt_id):
    output = {}
    total = 0
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        i = 0
        for line in input_file:
            tokens = line.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2].strip()
            score = float(tokens[6])
            
            to_lower = True
            
            if essay_set == prompt_id or prompt_id <= 0:
                
                if to_lower:
                
                    content = content.lower()

                    content_tokenized = tokenize(content)
                    
                    content_sents = nltk.sent_tokenize(content)
                    
                    sents = []
                    
                    for sent in content_sents:
                        
                        sents.append(tokenize(sent))
                    
                if params['remove_stopwords']:
                    
                    content_tokenized, sents = filter_stopwords(content_tokenized,sents)
                    
                if params['padding_level']=='document':
                        
                    total += len(content_tokenized)
                    
                else:
                    
                    for sent in sents:
                        
                        total += len(sent)
                
                output[i]={'set':essay_set,
                      'content':content_tokenized,
                      'sents':sents,
                      'score':score,
                      'essay_id':essay_id}
                i += 1
    
    return output, total


def load_fold(train_path,dev_path,test_path,prompt):
    
    train_essays,train_num_words = load(train_path,prompt_id=prompt)
    
    dev_essays,_ = load(dev_path,prompt_id=prompt)
    
    test_essays,_ = load(test_path,prompt_id=prompt)
    
    logger.info("len_train_essays: %d, train_total_words: %d"%(len(train_essays),train_num_words))
    
    logger.info( "len_dev_essays:%d"%len(dev_essays))
    
    logger.info( "len_test_essays:%d"%len(test_essays))
    
    logger.info( "#Essays: %d"%(len(train_essays)+ len(dev_essays)+len(test_essays)))
    
    return train_essays,dev_essays,test_essays



def prepare_data(data_essays):
    data_ids = []
    data_x = []
    data_y = []
    data_set = []
    sorted_keys = sorted(data_essays.keys())
    for k in sorted_keys:
        v = data_essays[k]
        data_ids.append(v['essay_id'])
        if params['padding_level'] == 'document':
            data_x.append(v['content'])
        elif params['padding_level'] == 'sentence':
            data_x.append(v['sents'])
        data_y.append(v['score'])
        data_set.append(v['set'])
    return (data_x,data_y,data_ids,data_set)


ref_scores_dtype = 'int32'

asap_ranges = {
	0: (0, 60),
	1: (2,12),
	2: (1,6),
	3: (0,3),
	4: (0,3),
	5: (0,4),
	6: (0,4),
	7: (0,30),
	8: (0,60)
}

def get_ref_dtype():
	return ref_scores_dtype

def get_score_range(prompt_id):
	return asap_ranges[prompt_id]

def get_model_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = (scores_array - low) / (high - low)
	else:
		assert scores_array.shape[0] == prompt_id_array.shape[0]
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = (scores_array - low) / (high - low)
	assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
	return scores_array

def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
	arg_type = type(prompt_id_array)
	assert arg_type in {int, np.ndarray}
	if arg_type is int:
		low, high = asap_ranges[prompt_id_array]
		scores_array = scores_array * (high - low) + low
		assert np.all(scores_array >= low) and np.all(scores_array <= high)
	else:
		assert scores_array.shape[0] == prompt_id_array.shape[0]
		dim = scores_array.shape[0]
		low = np.zeros(dim)
		high = np.zeros(dim)
		for ii in range(dim):
			low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
		scores_array = scores_array * (high - low) + low
	return scores_array

##################
## import lang
##################
from lang import Lang

def make_lang(train_x,dev_x,test_x):
    
    lang = Lang('essays')
    
    train_words = []
    
    for essay in train_x:
        
        if params['padding_level']=='document':
            
            train_words += essay
            
        elif params['padding_level']=='sentence':
            
            for sent in essay:
                
                train_words += sent
    
    lang.build_vocab(train_words, voc_size=params['voc_size'])
    
    params['voc_size'] = lang.n_words

    lang.make_embeddings(emb_size=params['emb_size'], emb_type=params['emb_type'])
    
    return lang

def get_MAX_LENS(train_x, dev_x, test_x):
    
    if params['padding_level'] == 'document':
        train_lens = [len(e) for e in train_x]
        dev_lens = [len(e) for e in dev_x]
        test_lens = [len(e) for e in test_x]
    
        params['max_train_len'] = np.max(train_lens)
        params['max_dev_len'] = np.max(dev_lens)
        params['max_test_len'] =  np.max(test_lens)

    
        params['max_doc_len'] = \
        np.max([params['max_train_len'],
                params['max_dev_len'],
                params['max_test_len']])
    
    elif params['padding_level'] == 'sentence':
        
        train_lens = [len(e) for e in train_x] # ns
        dev_lens = [len(e) for e in dev_x]# ns
        test_lens = [len(e) for e in test_x]# ns
        
        
    
        params['max_train_len'] = np.max(train_lens)# max ns
        params['max_dev_len'] = np.max(dev_lens)# max ns
        params['max_test_len'] =  np.max(test_lens)# max ns

    
        params['max_doc_len'] = \
        np.max([params['max_train_len'],
                params['max_dev_len'],
                params['max_test_len']]) # max ns in whole corpus
    
        max_sent_len = 0
        
        for essay in train_x + dev_x + test_x:
            
            for sent in essay:
                
                if len(sent) > max_sent_len:
                
                    max_sent_len = len(sent)
       
        params['max_sent_len'] = max_sent_len
            

def convert_seq_to_indices(data_x, lang, max_seq_len):
    
    if params['padding_level'] == 'document':
        
        output_x = []
        output_mask = []
        
        unk_hit = 0.
        num_hit =0.
        total = 0.
        
        for seq in data_x:
            out_seq,out_mask,(unk,num,tot) =  lang.indicies_from_sentence(seq ,max_seq_len, params['padding_place']) # does padding and indexing
            output_x.append(out_seq)
            output_mask.append(out_mask)
            unk_hit += unk
            num_hit += num
            total += tot
            
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    
    elif params['padding_level'] == 'sentence':

        output_x = []
        output_mask = []
        
        unk_hit = 0.
        num_hit =0.
        total = 0.
        
        for essay in data_x:
            
            essay_output = []
            essay_mask = []
            
            for seq in essay:
                out_seq,out_mask,(unk,num,tot) = \
                    lang.indicies_from_sentence(seq, max_seq_len,params['padding_place']) # do padding and indexing
                
                unk_hit += unk
                num_hit += num
                total += tot
                essay_output.append(out_seq)
                essay_mask.append(out_mask)
                
            output_x.append(essay_output)
            output_mask.append(essay_mask)
            
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))

    return output_x, output_mask
        
##################
### Models
##################
from models_masking import Model

########################        
#### Batching
########################

####################
## without sorting
###################
def make_batch_indices(size, batch_size):
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(num_batches)]


def get_batches(data_samples, #data_x,data_mask,data_y,data_pmt,data_scores, feat
                batch_size):
    
    n = len(data_samples)

    batch_indices = make_batch_indices(n,batch_size)

    data_batches = [data_samples[start:end] for (start,end) in batch_indices]
    
    batches = []

    for i, batch in enumerate(data_batches):
        
        batch_seq_len = []
                        
        batch_text = []
        
        batch_label  = []
        
        batch_pmt = []
        
        batch_score = []
        
        batch_mask = []
        
        batch_feat = []
        
        for sample in batch:
            
            batch_text.append(sample[0])
            
            batch_mask.append(sample[1])
           
            batch_seq_len.append(np.sum(sample[1]))
            
            batch_label.append(sample[2])
            
            batch_pmt.append(sample[3])
            
            batch_score.append(sample[4])
            
            batch_feat.append(sample[5])
        
 
        
        batches.append((batch_text,
                        batch_label,
                        batch_seq_len,
                        batch_mask,
                        batch_pmt,
                        batch_score,
                        batch_feat))

    return batches



#%%
##################
### Evaluation
##################
def predict(data_samples, model, batch_size, lang, max_len):
    
    predictions = []
    
    batches = get_batches(data_samples, batch_size)

    pmt = []
    
    score = []
    
    for bi,batch in enumerate(batches):
        
        inp, label, seq_lens, mask, batch_pmt, batch_score, feat =  \
         batch[0], batch[1] , batch[2], batch[3] , batch[4], batch[5] ,batch[6] 

        input_var = Variable(torch.LongTensor(inp))
        label_var = Variable(torch.FloatTensor(label))
        seq_lens_var = Variable(torch.FloatTensor(seq_lens))
        mask_var = Variable(torch.FloatTensor(mask))
        feat_var = Variable(torch.FloatTensor(feat))
        
        if params['RUN_ON_GPU'] and cuda.is_available():
            input_var = input_var.cuda()
            label_var = label_var.cuda()
            mask_var = mask_var.cuda()
            seq_lens_var = seq_lens_var.cuda()
            feat_var = feat_var.cuda()
            
        coh_score = model(input_var, seq_lens_var, batch_size, mask_var, feat_var)
        
        predictions.append(coh_score.data)
        
        pmt += batch_pmt
        
        score += batch_score
    

    predictions = torch.stack(predictions,dim=0)
    
    predictions =  predictions.view(-1)
    

    if params['RUN_ON_GPU']:
        
        predictions =  predictions.cpu().numpy()
        
    else:
        
        predictions =  predictions.numpy()


    promt_array = np.array(pmt)
    
    score_array = np.array(score)
    
    pred = convert_to_dataset_friendly_scores(predictions, promt_array)

    return predictions, pred , score_array


from nea.my_kappa_calculator import quadratic_weighted_kappa as qwk

def calc_qwk(pred, scores, low, high):
    
    # Kappa only supports integer values
    pred_int = np.rint(pred).astype('int32')

    scores =  scores.astype('int32')
    
    pred_qwk = qwk(pred_int,scores, low, high)
    
    return pred_qwk


def eval(model, data_samples, prompt, batch_size, lang,max_len):
    
    model.eval()

    model_friendly_predictions, dataset_friendly_pred, scores =     \
    predict(data_samples, model, batch_size, lang,max_len)
        
    low, high = get_score_range(prompt)

    qwk = calc_qwk(dataset_friendly_pred, scores,  low, high)
    
    return qwk


###################
## Train function
###################
from torch.autograd import Variable
import math
def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))

#%%
def train(trainset_samples, # train_x,train_mask,train_y,train_pmt,train_scores
          devset_samples,
          testset_samples,
          model,
          prompt,
          batch_size,
          lang,
          shuffle=True):
       
    optimizer = torch.optim.Adam(model.parameters(), eps=1e-06, lr=params['lr'])
    
    num_epochs = params['num_epochs']
    
    steps = 0
    
    qwk_train = 0.0
    
    qwk_dev = 0.0
    
    hist_epoch = []
    
    hist_loss = []
    
    hist_qwk_train = []
    
    hist_qwk_dev = []
    
    best_qwk_dev = 0.0
    
    qwk_test = 0.0  
    
    for epoch in range(num_epochs): 
        
        if shuffle:
            random.shuffle(trainset_samples)
        
        train_batches = get_batches(trainset_samples,
                            batch_size)

        epoch_loss = 0.0
        
        for bi, train_batch in enumerate(train_batches):
         
            inp, label, seq_lens, mask, feat =  train_batch[0],train_batch[1],train_batch[2],train_batch[3],train_batch[6]

            input_var = Variable(torch.LongTensor(inp))
            label_var = Variable(torch.FloatTensor(label))
            seq_lens_var = Variable(torch.FloatTensor(seq_lens))
            mask_var = Variable(torch.FloatTensor(mask))
            feat_var = Variable(torch.FloatTensor(feat))
            
            
            target_var = input_var[:,1:]
            eos_var = Variable(torch.LongTensor([lang.EOS_index]*input_var.size()[0]))
            eos_var = eos_var.unsqueeze(-1)
            target_var = torch.cat([target_var,eos_var],dim=1)

            if params['RUN_ON_GPU'] and cuda.is_available():
                input_var = input_var.cuda()
                label_var = label_var.cuda()
                seq_lens_var = seq_lens_var.cuda()
                mask_var = mask_var.cuda()
                target_var = target_var.cuda()
                feat_var = feat_var.cuda()
            
            model.train() # set the model's mode to training for dropout
            
            optimizer.zero_grad()
                        
            coh_score = model(input_var, seq_lens_var, batch_size, mask_var, feat_var)
            
            loss_coh = F.mse_loss(coh_score, label_var)
                
            loss = loss_coh
            
            loss.backward()
            
            # clip_norm_2            
            torch.nn.utils.clip_grad_norm(model.parameters(),params['clip'])
            
            #clip_value
#            parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
#            for p in parameters:
#                p.grad.data.clamp_(max=0)
            
            optimizer.step()

            epoch_loss += loss.data[0]
            
            steps += 1
            
            if params['run_eval']:
            
                if (steps-1) % len(train_batches) == 0:
                    qwk_dev =  eval(model, devset_samples,
                                    prompt, batch_size= len(devset_samples) ,#180,
                                    lang=lang,
                                    max_len = params['max_doc_len'])
                    
                    hist_qwk_dev.append(qwk_dev)
                    
                    if qwk_dev >= best_qwk_dev:
                        
                        best_qwk_dev = qwk_dev
                        
                        qwk_test =  eval(model, testset_samples,
                                         prompt, batch_size=len(testset_samples),#180,
                                         lang=lang,
                                         max_len = params['max_doc_len'])                    
                    
            
            if epoch % 1 ==0 and params['shell_print']:
                drawProgressBar(params['shell_print'],'epoch:%d, '%(epoch+1),bi+1, len(train_batches),
                                ' loss:%.4f,  qwk_train:%.3f, qwk_dev:%.3f qwk_test:%.3f'\
                                %(epoch_loss/float(bi+1), qwk_train, qwk_dev, qwk_test))
            
                
        # end of batches
        if epoch % 1 ==0 and params['shell_print']:
            
            print "\n"
        
        if epoch % 1 ==0 and params['shell_print']==False:
            print 'epoch:%d, '%(epoch+1),bi+1, len(train_batches),\
                                ' loss:%.4f,  qwk_train:%.3f, qwk_dev:%.3f qwk_test:%.3f'\
                                %(epoch_loss/float(bi+1), qwk_train, qwk_dev, qwk_test)
        
        hist_epoch.append(epoch)
        
        hist_loss.append(epoch_loss/len(train_batches))
        
    return hist_epoch,hist_loss,hist_qwk_train,hist_qwk_dev, qwk_test
#%%
def doc_padding_by_sent_pad(x, mask, lang):

    sent_pad, out_mask, _ =  lang.indicies_from_sentence([], 
                                                         params['max_sent_len'], 
                                                         params['padding_place'])
    
    data_x = []
    data_mask =[]
    
    for es, mask in zip(x, mask):
        doc_pad = [sent_pad] * params['max_doc_len']
        if params['padding_place']=='pre':
            doc_pad[-len(es):] = es
        elif params['padding_place']=='post':
            doc_pad[:len(es)] = es
        doc_pad_flat = []
        for sent in doc_pad:
            doc_pad_flat += sent
        data_x.append(doc_pad_flat)
        doc_mask = [out_mask] * params['max_doc_len']
        if params['padding_place']=='pre':
            doc_mask[-len(mask):] = mask
        elif params['padding_place']== 'post':
            doc_mask[:len(mask)] = mask
        doc_mask_flat = []
        for m in doc_mask:
            doc_mask_flat += m
        data_mask.append(doc_mask_flat)

    return data_x, data_mask


#%%
import pickle
from sklearn import preprocessing
def main(fold_path, prompt):
    
    # get data
    train_path = fold_path+'/train.tsv'
    dev_path = fold_path+'/dev.tsv'
    test_path = fold_path+'/test.tsv'
    
    train_essays, dev_essays, test_essays = \
            load_fold(train_path, dev_path, test_path, prompt)
        
    (train_x,train_scores,train_ids,train_pmt) = prepare_data(train_essays)
    (dev_x,dev_scores,dev_ids,dev_pmt) = prepare_data(dev_essays)
    (test_x,test_scores,test_ids,test_pmt) = prepare_data(test_essays)

    with open(fold_path+'/train_feat_p%s.pkl'%(str(prompt)),'rb') as f:
        train_feat = pickle.load(f)
        
    with open(fold_path+'/dev_feat_p%s.pkl'%(str(prompt)),'rb') as f:
        dev_feat = pickle.load(f)
        
    with open(fold_path+'/test_feat_p%s.pkl'%(str(prompt)),'rb') as f:
        test_feat = pickle.load(f)

    
    logger.info(" train_feat: %s, dev_feat: %s, test_feat: %s"%(str(train_feat.shape), str(dev_feat.shape), str(test_feat.shape)))
    
    train_feat =  preprocessing.scale(train_feat,axis=0)
    dev_feat =  preprocessing.scale(dev_feat,axis=0)
    test_feat =  preprocessing.scale(test_feat,axis=0)

    
    train_feat = list(train_feat)
    dev_feat = list(dev_feat)
    test_feat = list(test_feat)

    
    lang = make_lang(train_x,dev_x,test_x)
    

#    model_friendly_scores = get_model_friendly_scores(np.array(test_scores), np.array(test_pmt))
#
#    for i in range(len(test_x)):
#        essay = test_x[i]
#        model_friendly_score =model_friendly_scores[i]
#        if model_friendly_score > 0.0:
#            print
#            print test_ids[i]
#            print essay
#            print model_friendly_score
#            var = raw_input("Do you want to see the next essay (Y/N): ")
#            if var.lower() == 'n':
#                break 
#    return

    get_MAX_LENS(train_x,dev_x,test_x)
    
    if params['padding_level'] == 'document':
        logger.info( "train padding (document level) and indexing  ....")
        train_x, train_mask = convert_seq_to_indices(train_x, lang, params['max_doc_len'])
        logger.info( "dev padding (document level) and indexing ....")
        dev_x, dev_mask = convert_seq_to_indices(dev_x, lang, params['max_doc_len'])
        logger.info( "test padding (document level) and indexing ....")
        test_x, test_mask = convert_seq_to_indices(test_x, lang, params['max_doc_len'])
    
    elif params['padding_level'] == 'sentence':
        logger.info( "train padding (sentence level) and indexing ....")
        train_x, train_mask = convert_seq_to_indices(train_x, lang, params['max_sent_len'])
        logger.info( "dev padding (sentence level) and indexing ....")
        dev_x, dev_mask = convert_seq_to_indices(dev_x, lang, params['max_sent_len'])
        logger.info( "test padding (sentence level) and indexing ....")
        test_x, test_mask = convert_seq_to_indices(test_x, lang, params['max_sent_len'])
        
        logger.info( "make length of all sentences equal ....")
        sent_pad, out_mask, _ =  lang.indicies_from_sentence([], params['max_sent_len'],params['padding_place'])

        train_x, train_mask =  doc_padding_by_sent_pad(train_x, train_mask, lang)
        dev_x, dev_mask =  doc_padding_by_sent_pad(dev_x, dev_mask, lang)
        test_x, test_mask =  doc_padding_by_sent_pad(test_x, test_mask, lang)

        
    logger.info("prompt:%d"%prompt)
    
    logger.info("score: %d - %d"%(np.min(train_scores+dev_scores+test_scores),
                            np.max(train_scores+dev_scores+test_scores)))
    
    logger.info("score Med: %d"%(np.median(train_scores+dev_scores+test_scores)))
    
    train_scores = np.array(train_scores,dtype = 'float')
    dev_scores = np.array(dev_scores,dtype = 'float')
    test_scores = np.array(test_scores,dtype = 'float')
    
    train_pmt = np.array(train_pmt, dtype='int32')
    dev_pmt = np.array(dev_pmt, dtype='int32')
    test_pmt = np.array(test_pmt, dtype='int32')
    
    
    train_mean = train_scores.mean(axis=0)
    train_std = train_scores.std(axis=0)
    dev_mean = dev_scores.mean(axis=0)
    dev_std = dev_scores.std(axis=0)
    test_mean = test_scores.mean(axis=0)
    test_std = test_scores.std(axis=0)
    
    logger.info( "train_mean = %f , train_std: %f"%(train_mean , train_std))
    logger.info( "dev_mean %f , dev_std:%f"%(dev_mean , dev_std))
    logger.info( "dev_mean %f , dev_std:%f"%(test_mean , test_std))
    
    
    # Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
    train_y = dataset.get_model_friendly_scores(train_scores, train_pmt)
    dev_y = dataset.get_model_friendly_scores(dev_scores, dev_pmt)
    test_y = dataset.get_model_friendly_scores(test_scores, test_pmt)

    
    train_samples = zip(train_x,train_mask,train_y,train_pmt,train_scores,train_feat)
    dev_samples = zip(dev_x,dev_mask,dev_y, dev_pmt, dev_scores,dev_feat)
    test_samples = zip(test_x,test_mask,test_y, test_pmt, test_scores,test_feat)
    
    
    ##########################
    ## model creation
    ##########################
    max_doc_len = params['max_doc_len']
    
    if params['padding_level']=='sentence':
        
         params['utt_size'] = params['max_sent_len']
         
    output_layer_size = train_feat[0].size # self.table == 4
         
    logger.info(" params:\n %s"%params)
    
    model = Model(max_doc_len=max_doc_len, 
                  relation_size=params['relation_size'],
                  lstm_size = params['lstm_size'], 
                  voc_size = params['voc_size'], 
                  emb_size=params['emb_size'], 
                  dropout_rate=params['dropout_rate'],
                  embeddings = lang.embeddings,
                  mean_y = train_y.mean(axis=0),
                  pad_idx = lang.PAD_index,
                  utt_size =  params['utt_size'],
                  mode = params['model'],
                  table = params['result_table'],
                  output_layer_size = output_layer_size)
    
    
    if params['RUN_ON_GPU'] and cuda.is_available():
        model = model.cuda()

    #############################
    ## start training
    #############################    
    if params['RUN_ON_GPU'] and cuda.is_available():    
        hist_epoch,hist_loss,hist_qwk_train,hist_qwk_dev, qwk_test = \
                                train(trainset_samples=train_samples,
                                     devset_samples=dev_samples,
                                     testset_samples = test_samples,
                                     model=model,
                                     prompt=prompt,
                                     batch_size=params['batch_size'],
                                     lang=lang)
                                
    else:
        hist_epoch,hist_loss,hist_qwk_train,hist_qwk_dev, qwk_test = \
                            train(trainset_samples = train_samples[:10],
                                 devset_samples = train_samples[:10],
                                 testset_samples = test_samples[:10],
                                 model=model,
                                 prompt=prompt,
                                 batch_size=params['batch_size'],
                                 lang=lang)
                            
                            
    return  qwk_test, model
                

#%%
    
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', required=True, help="Promp ID for ASAP dataset. ")
    
    parser.add_argument("-f", "--fold", dest="fold_id", type=str, metavar='<str>', required=True, help="The ID of the fold.")
    
    parser.add_argument("-m", "--model", dest="model", type=str, metavar='<str>', required=True, help="Model type:coh_emb|coh_lstm|lstm")
    
    parser.add_argument("-t", "--table", dest="table", type=float, metavar='<float>', required=True, help="Table ID (coh_output.lstm_output) in results: 1.0|2.0|3.0")
    
    parser.add_argument("-rs", "--remove_stopword", dest="remove_stopword", type=str, metavar='<str>', required=True, help="Remove stopWords")
    
    
    
    args = parser.parse_args()
    
    params['model'] = args.model
    
    params['result_table'] = args.table
    
    if args.remove_stopword.lower() == 'true':
        
        params['remove_stopwords'] = True
    
    elif args.remove_stopword.lower() == 'false':
    
        params['remove_stopwords'] = False
    
    else:
    
        raise NotImplemented
         

    prompt = args.prompt_id
    
    fold_path = "./data/fold_" + args.fold_id
    
    qwk_test, model = main(fold_path=fold_path, prompt=prompt)
    
    print "\n*********************"
    print "** qwk_test: %.3f ***"%(qwk_test)
    print "********************\n"
            

        

    
        
        
    
