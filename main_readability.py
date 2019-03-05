# coding: utf-8
#%%
from params_readability import params

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if params['RUN_ON_GPU']:
    import torch.cuda as cuda

import warnings
warnings.filterwarnings('ignore')

import random
import torch
random.seed(params['seed'])
torch.manual_seed(params['seed'])

import torch.nn.functional as F


import numpy as np 
from utility import drawProgressBar

import nltk
import re

num_regex = re.compile(u'^[+-]?[0-9]+[\,\.]?[0-9]*$')

#%%
def tokenize(string):

    tokens = nltk.word_tokenize(string)

    for index, token in enumerate(tokens):

        if token == '@' and (index+1) < len(tokens):

            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])

            tokens.pop(index)    

    return tokens

def is_number(token):
    
    return bool(num_regex.match(token))

#%%
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

stopwords += [u'``',u"''",u"lt",u"gt", u"<NUM>"]

def filter_stopwords(sents):

    filterd_sents = []
        
    for sent in sents:
    
        filterd_sent = []
        
        for word in sent:
        
            if word not in stopwords:
            
                filterd_sent.append(word)
        
        filterd_sents.append(filterd_sent)

    return filterd_sents
#%%

import pickle

def load(file_path):
    
    with open(file_path,'rb') as f:
        
        dataset = pickle.load(f)
    
    texts = dataset[0]
    
    samples = dataset[1]
    
    num_hit =0
    total= 0
    
    processed_texts = []
    
    for text in texts:
    
        text = text.strip()
        
        text = text.lower()
        
        sents = nltk.sent_tokenize(text)
        
        out_sents = []
        
        for sent in sents:
            
            if len(sent)>0:
            
                sent = sent.strip()
                
                sent = tokenize(sent)
                
                sent_filtered = []
                
                for word in sent:
                
                    word = word.strip()
                    
                    if is_number(word):
                    
                        sent_filtered.append(u'<NUM>')
                        
                        num_hit += 1
                    
                    else:
                    
                        sent_filtered.append(word)
                    
                    total += 1
                
                out_sents.append(sent_filtered)
                
        if params['remove_stopwords']:
            
             out_sents = filter_stopwords(out_sents)        
        
        processed_texts.append(out_sents)
        
    output = (processed_texts, samples)
    
    logger.info(' <num> hit rate: %.2f%%' % (100*num_hit/total))
    
    return output

#%%
def get_five_folds(samples):
    
    indices = range(len(samples))
    
    random.shuffle(indices)
    
    fold_size = int(len(indices) / 10)
    
    indices = indices[:10*fold_size]
    
    folds = []
    
    for k in range(10):
       
        fold = indices[k*fold_size:(k+1)*fold_size]
        
        folds.append(fold)
    
    
    output = []
    for i,fold in enumerate(folds):
        
        fold_test = fold
        
        fold_train_dev =[]
        
        for j,f in enumerate(folds):
            
            if j!=i:
                
                fold_train_dev += f
        
        n = len(fold_train_dev)
        
        train_ind = fold_train_dev[:int(0.8*n)]
        
        dev_ind = fold_train_dev[int(0.8*n):]
        
        test_ind = fold_test
        
        train_set = [samples[ind] for ind in train_ind]
        dev_set = [samples[ind] for ind in dev_ind]
        test_set = [samples[ind] for ind in test_ind]
        
        output.append([train_set,dev_set,test_set])
        
    return output

def prepare_data(data_set, texts):
    
    data_x = []
    
    data_y = []
    
    zero_lable_count = 0.
    
    for sample in data_set:
        
        t0 = texts[int(sample[0])]
        
        t1 = texts[int(sample[1])]
        
        l =  int(sample[2])
        
        if l == 1:
            
            label = 1
            
            data_x.append((t0,t1))
            
            data_y.append(label)
            
        elif l == -1: 
            
            label = 1

            data_x.append((t1, t0))
            
            data_y.append(label)
        
        else:
        
            zero_lable_count+=1
            
    logger.info(" zero_label count: %.2f, hit rate: %.2f%%"%(zero_lable_count,zero_lable_count/len(data_set)))
    
    logger.info(" zero labels are filterd out to have a binary classifcation")
    
    return (data_x, data_y) # ([(t0,t1),(t0,t1),(t0,t1)],[y0,y1,y2])

#%%
##################
## import lang
##################
from lang import Lang

def make_lang(texts):
    
    lang = Lang('texts')
    
    words = []
    
    for text in texts:
    
        for sent in text:
        
            words += sent
    
    lang.build_vocab(words, voc_size=params['voc_size'])
    
    params['voc_size'] = lang.n_words
    
    lang.make_embeddings(emb_size=params['emb_size'], 
                         emb_type=params['emb_type'])

    return lang
#%%
def get_MAX_LENS(texts):

    sent_lens_list = []
    
    doc_lens_list = []
    
    for text in texts:
    
        doc_lens_list.append(len(text))
        
        for sent in text:
            
            sent_lens_list.append(len(sent))
            
    params['max_doc_len'] = np.max(doc_lens_list)
    params['max_sent_len'] = np.max(sent_lens_list)
#%%                
def convert_seq_to_indices(data_x, lang, max_seq_len):

        output_x = []
        output_mask = []
        
        unk_hit = 0.
        num_hit =0.
        total = 0.
        
        for text0, text1 in data_x:
            
            text0_output = []
            text0_mask = []

            text1_output = []
            text1_mask = []

            for seq in text0:
                
                out_seq, out_mask,(unk, num,tot) = \
                    lang.indicies_from_sentence(seq, 
                                                max_seq_len,
                                                params['padding_place']) # do padding and indexing
                
                unk_hit += unk
                num_hit += num
                total += tot
                text0_output.append(out_seq)
                text0_mask.append(out_mask)
                
            
            
            for seq in text1:
                
                out_seq, out_mask,(unk, num,tot) = \
                    lang.indicies_from_sentence(seq, 
                                                max_seq_len,
                                                params['padding_place']) # do padding and indexing
                
                unk_hit += unk
                num_hit += num
                total += tot
                text1_output.append(out_seq)
                text1_mask.append(out_mask)
                
            output_x.append((text0_output,text1_output))
            output_mask.append((text0_mask,text1_mask))
            
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))

        return output_x, output_mask
    
#%%     
from models_ra import Model
#%%

########################        
#### Batching
########################

def make_batch_indices(size, batch_size):
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(num_batches)]


def get_batches(data_samples, # [(text0,text1), (mask0, mask1), label]
                batch_size):
    
    n = len(data_samples)

    batch_indices = make_batch_indices(n, batch_size)

    data_batches = [data_samples[start:end] for (start,end) in batch_indices]
    
    batches = []

    for i, batch in enumerate(data_batches):
        
        batch_texts0 = []
        
        batch_texts1 = []
        
        batch_masks0 = [] 
        
        batch_seq_lens0 = []
        
        batch_masks1 = []
        
        batch_seq_lens1 = []
        
        batch_labels = []
        
        for sample in batch:
            
            docs = sample[0]
            
            masks = sample[1]
            
            label = sample[2]
           
            d0 = docs[0]

            d1 = docs[1]
            
            mask0 = masks[0]
            
            mask1 = masks[1]
    
            seq_lens0 = np.sum(mask0)
            
            seq_lens1 = np.sum(mask1)
            
            batch_texts0.append(d0)
            
            batch_texts1.append(d1)
            
            batch_masks0.append(mask0)
            
            batch_masks1.append(mask1)
            
            batch_seq_lens0.append(seq_lens0)
            
            batch_seq_lens1.append(seq_lens1)
            
            batch_labels.append(label)
    
        batches.append((batch_texts0, batch_texts1, batch_masks0, batch_masks1,batch_seq_lens0, batch_seq_lens1, batch_labels))
        
    return batches

#%%
def predict(data_samples, model, batch_size, lang, max_len):
    
    score0 = []
    score1 = []
    
    
    batches = get_batches(data_samples, batch_size)
        
    for bi, batch in enumerate(batches):
         
        # batch_texts0, batch_texts1, batch_masks0, batch_masks1, batch_labels
        input0, input1, mask0, mask1, seq_lens0, seq_lens1, label =\
        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5] , batch[6] 
            
            
        input0_var = Variable(torch.LongTensor(input0))
        input1_var = Variable(torch.LongTensor(input1))
            
        mask0_var = Variable(torch.FloatTensor(mask0))
        mask1_var = Variable(torch.FloatTensor(mask1))
        
        seq_lens0_var = Variable(torch.FloatTensor(seq_lens0))
        seq_lens1_var = Variable(torch.FloatTensor(seq_lens1))
       
        label_var = Variable(torch.FloatTensor(label))
            
        if params['RUN_ON_GPU'] and cuda.is_available():
            
            input0_var = input0_var.cuda()
            input1_var = input1_var.cuda()
            
            mask0_var = mask0_var.cuda()
            mask1_var = mask1_var.cuda()
            
            seq_lens0_var = seq_lens0_var.cuda()
            seq_lens1_var = seq_lens1_var.cuda()
            
            label_var = label_var.cuda()
                
        coh_score0 = model(input0_var, seq_lens0_var, batch_size, mask0_var, None)
        
        coh_score1 = model(input1_var, seq_lens1_var, batch_size, mask1_var, None)
    
        score0.append(coh_score0.cpu().data)
        
        score1.append(coh_score1.cpu().data)
    
    score0 = torch.cat(score0, dim=0)
    
    score1 = torch.cat(score1, dim=0)
   
    return (score0, score1)
    
#%%  
def eval(model, data_samples, batch_size, lang, max_len):
    
    model.eval()
    
    scores =  predict(data_samples, model, batch_size, lang, max_len)
        
    coh_score0  = scores[0]
        
    coh_score1  = scores[1]

    diff = coh_score0 -  coh_score1
    
    diff = F.relu(diff)

    diff[diff!=0]=1.
    
    diff = diff.cpu().data.numpy()
    
    # compute acc
    correct = np.sum(diff)
    total = float(len(diff))
    acc = correct/ total
    return acc

#%%
from torch.autograd import Variable
###################
# ## Train
###################
def train(trainset_samples,
          devset_samples,
          testset_samples,
          model, batch_size, lang, shuffle=True):
    
    optimizer = torch.optim.Adam(model.parameters(), eps=1e-06, lr=params['lr'])
    
    num_epochs = params['num_epochs']
    
    steps = 0
    
    acc_dev = 0.0
        
    best_acc_dev = 0.0
    
    acc_test = 0.0
    
    for epoch in range(num_epochs): 
        
        if shuffle:
            
            random.shuffle(trainset_samples)
        
        
        train_batches = get_batches(trainset_samples,
                            batch_size)
        
        epoch_loss = 0.0
        
        for bi, train_batch in enumerate(train_batches):
         
            # batch_texts0, batch_texts1, batch_masks0, batch_masks1, batch_labels
            input0, input1, mask0, mask1, seq_lens0, seq_lens1, label =\
            train_batch[0], train_batch[1], train_batch[2], train_batch[3], train_batch[4], train_batch[5] , train_batch[6] 
            
            
            input0_var = Variable(torch.LongTensor(input0))
            input1_var = Variable(torch.LongTensor(input1))
            
            mask0_var = Variable(torch.FloatTensor(mask0))
            mask1_var = Variable(torch.FloatTensor(mask1))
            
            seq_lens0_var = Variable(torch.FloatTensor(seq_lens0))
            seq_lens1_var = Variable(torch.FloatTensor(seq_lens1))
           
            label_var = Variable(torch.FloatTensor(label))
            
            
            
            if params['RUN_ON_GPU'] and cuda.is_available():
                
                input0_var = input0_var.cuda()
                input1_var = input1_var.cuda()
                
                mask0_var = mask0_var.cuda()
                mask1_var = mask1_var.cuda()
                
                seq_lens0_var = seq_lens0_var.cuda()
                seq_lens1_var = seq_lens1_var.cuda()
                
                label_var = label_var.cuda()
            
            
            optimizer.zero_grad()
            
            coh_score0 = model(input0_var, seq_lens0_var, batch_size, mask0_var, None)
            
            coh_score1 = model(input1_var, seq_lens1_var, batch_size, mask1_var, None)
           
            loss = torch.nn.MarginRankingLoss(margin=1)(coh_score0,coh_score1,label_var)
            
            loss.backward()
            
            #torch.nn.utils.clip_grad_norm(model.parameters(),params['clip'])
            
            optimizer.step()

            epoch_loss += loss.data[0]
            
            steps += 1
            
            if params['run_eval']:
                        
                if (steps-1) % len(train_batches) == 0:
                    
                    acc_dev =  eval(model, devset_samples,
                                    batch_size=params['test_batch_size'],
                                    lang=lang,
                                    max_len = params['max_doc_len'])
                    
                   
                    
                    if acc_dev >= best_acc_dev:
                        
                        best_acc_dev = acc_dev
                        
                        acc_test =  eval(model, testset_samples,
                                          batch_size= params['test_batch_size'], #len(testset_samples),
                                          lang=lang,
                                          max_len = params['max_doc_len'])
                    
                    
            
            if epoch % 1 ==0:
                drawProgressBar(params['shell_print'],'epoch:%d, '%(epoch+1),bi+1, len(train_batches),
                                ' loss:%.4f,  acc_dev:%.4f acc_test:%.4f'\
                                %(epoch_loss/float(bi+1), acc_dev, acc_test))
        # end of batches
        if epoch % 1 ==0:
            
            print "\n"       

        
    return acc_test
#%%
def doc_padding_by_sent_pad(x, mask, lang):

    sent_pad, out_mask, _ =  lang.indicies_from_sentence([], 
       
                                                         params['max_sent_len'], 
                                                         params['padding_place'])
    data_x = []
    
    data_mask =[]
    
    n = len(x)
    
    for i in range(n):
        
        txt_pair = x[i]
        
        mask_pair = mask[i] 
        
        t0 =  txt_pair[0]
        
        t1 =  txt_pair[1]
    
        m0  = mask_pair[0]
        
        m1 = mask_pair[1]
        
        ###
        
        t0_pad = [sent_pad] * params['max_doc_len']
        
        t0_pad[:len(t0)] = t0
    
        t0_pad_flat = []
            
        for sent in t0_pad:
        
            t0_pad_flat += sent
        
        t0_mask = [out_mask] * params['max_doc_len']
        
        t0_mask[:len(m0)] = m0
        
        t0_mask_flat = []
        
        for m in t0_mask:
            
                t0_mask_flat += m
 
        ### 
        
        t1_pad = [sent_pad] * params['max_doc_len']
        
        t1_pad[:len(t1)] = t1
    
        t1_pad_flat = []
            
        for sent in t1_pad:
        
            t1_pad_flat += sent
        
        t1_mask = [out_mask] * params['max_doc_len']
        
        t1_mask[:len(m1)] = m1
        
        t1_mask_flat = []
        
        for m in t1_mask:
            
                t1_mask_flat += m
  
        data_x.append((t0_pad_flat, t1_pad_flat))
        
        data_mask.append((t0_mask_flat, t1_mask_flat))

    return data_x, data_mask
   
#%%
def main(fold,texts):
    
    # fold = (train_set,dev_set, test_set)
    (train_set,dev_set, test_set) = fold    
    
    logger.info(" train_size: %d, dev_size: %d, test_size: %d"%(len(train_set), len(dev_set), len(test_set)))
    
    (train_x, train_y) = prepare_data(train_set,texts)
    
    (dev_x, dev_y) = prepare_data(dev_set, texts)
    
    (test_x, test_y) = prepare_data(test_set, texts)
    
    logger.info(" after zero-label filtering: train_size: %d, dev_size: %d, test_size: %d"%(len(train_x), len(dev_x), len(test_x)))
    
    lang = make_lang(texts)
     
    get_MAX_LENS(texts)
    
    logger.info( "train padding (sentence level) and indexing ....")
    train_x, train_mask = convert_seq_to_indices(train_x, lang, params['max_sent_len'])

    logger.info( "dev padding (sentence level) and indexing ....")
    dev_x, dev_mask = convert_seq_to_indices(dev_x, lang, params['max_sent_len'])
    
    logger.info( "test padding (sentence level) and indexing ....")
    test_x, test_mask = convert_seq_to_indices(test_x, lang, params['max_sent_len'])
    
    logger.info( "make length of all sentences equal ....")
    sent_pad, out_mask, _ =  lang.indicies_from_sentence([], params['max_sent_len'],params['padding_place'])

    train_x, train_mask = doc_padding_by_sent_pad(train_x, train_mask, lang)
    
    dev_x, dev_mask = doc_padding_by_sent_pad(dev_x, dev_mask, lang)
    
    test_x, test_mask = doc_padding_by_sent_pad(test_x, test_mask, lang)
        
    
    train_samples = zip(train_x,train_mask,train_y)
    
    dev_samples = zip(dev_x,dev_mask,dev_y)
    
    test_samples = zip(test_x,test_x,test_y)

    ##########################
    ## model creation
    ##########################
    
    max_doc_len = params['max_doc_len']
   
    params['utt_size'] = params['max_sent_len']
         
    logger.info(" params:\n %s"%params)
    
    model = Model(max_doc_len=max_doc_len, 
              relation_size=params['relation_size'],
              lstm_size = params['lstm_size'], 
              voc_size = params['voc_size'], 
              emb_size=params['emb_size'], 
              dropout_rate=params['dropout_rate'],
              embeddings = lang.embeddings,
              mean_y = None,
              pad_idx = lang.PAD_index,
              utt_size =  params['utt_size'],
              mode = params['model'],
              table = params['result_table'],
              output_layer_size = 0)
    
    
    if params['RUN_ON_GPU'] and cuda.is_available():
        model = model.cuda()
    
    #############################
    ## start training
    #############################    
    if params['RUN_ON_GPU'] and cuda.is_available():     
        acc_test = \
                    train(trainset_samples=train_samples,
                         devset_samples=dev_samples,
                         testset_samples = test_samples,
                         model=model,
                         batch_size=params['batch_size'],
                         lang=lang)
    else:
        acc_test = \
                    train(trainset_samples=train_samples[:10],
                         devset_samples=dev_samples[:10],
                         testset_samples = test_samples[:2],
                         model=model,
                         batch_size=params['batch_size'],
                         lang=lang)
                            
    return  acc_test
                

#%%
import argparse

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--fold", dest="fold_id", type=int, metavar='<int>', required=True, help="The ID of the fold.")    
    
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
         
    
    texts, samples = load('./de_clerq/dataset.pkl')
    
    
    folds = get_five_folds(samples)
    
    logger.info(" processing fold: %d"%args.fold_id)
    
    fold = folds[args.fold_id]
    
    acc_test = main(fold, texts)
    
    print "***********************"
    print "*** acc_test: %.2f%% **"%acc_test
    print "***********************"
    
