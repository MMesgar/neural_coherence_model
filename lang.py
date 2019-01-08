import torch
import torchtext.vocab as vocab
from torch.autograd import Variable
from nea import w2vEmbReader as load_emb
import pickle as pk
import logging
logger = logging.getLogger(__name__)


import operator
from collections import defaultdict


import re
num_regex = re.compile(u'^[+-]?[0-9]+\.?[0-9]*$')

class Lang:
    def __init__(self, name):
        self.name = name
        
        self.PAD_word = u'<PAD>'
        self.UNK_word = u'<UNK>'
        self.NUM_word = u'<NUM>'
        self.EOS_word = u'<EOS>'
        
        self.PAD_index = 0
        self.UNK_index = 1
        self.NUM_index = 2
        self.EOS_index = 3
        
        self.word2index = {}
        self.index2word = {}
  
        self.word2index[self.PAD_word] = self.PAD_index
        self.word2index[self.UNK_word] = self.UNK_index
        self.word2index[self.NUM_word] = self.NUM_index
        self.word2index[self.EOS_word] = self.EOS_index
        
        self.index2word[self.PAD_index] = self.PAD_word
        self.index2word[self.UNK_index] = self.UNK_word
        self.index2word[self.NUM_index] = self.NUM_word
        self.index2word[self.EOS_index] = self.EOS_word
        
        self.word2count = defaultdict(lambda: 0)
        
        self.n_words = len(self.index2word) # Count all words
        
        self.embeddings = None
        
    def build_vocab(self,data,voc_size):
      
        num_total_words = 0
        
        word2count = {}
        
        for word in data:
            
            num_total_words += 1
            
            if word not in word2count:
            
                word2count[word] = 1
            
            else:
            
                word2count[word] += 1

        if len(word2count)< voc_size:
            
            logger.error(" mismatch in num_voc in data (=%d) and voc_size (=%d)"%(len(word2count),voc_size))
            
        logger.info(" %d total words, %d unique words"%(num_total_words,len(word2count)))
        
        
        sorted_word_freqs = sorted(word2count.items(), 
                            key=operator.itemgetter(1),
                            reverse=True)
                   
        freq_vocabs= [w for w,_ in 
                            sorted_word_freqs[:(voc_size-len(self.word2index))]]
        

        #assert len(freq_vocabs)+len(self.word2index) == voc_size, "%d %d"%(len(freq_vocabs)+len(self.word2index),voc_size)
        
        for voc in freq_vocabs:
            self.word2index[voc] = self.n_words
            self.index2word[self.n_words] = voc
            self.word2count[voc] =  word2count[voc]
            self.n_words += 1

        #assert self.n_words == voc_size
        
                      
    def make_embeddings(self, emb_size, emb_type='random'):
        #initalize embed
        self.embeddings = torch.Tensor(self.n_words, emb_size).uniform_(-0.05, 0.05)
        logger.info(" initialize embeddings (size:(%d,%d)) by unifotm (-0.05, 0.05)"%(self.n_words, emb_size))
        
        if emb_type=='kaveh':
            pre_trained_emb = load_emb.W2VEmbReader('./En_vectors.txt',50)
            pre_trained_emb.vectors = torch.from_numpy(pre_trained_emb.vectors)
            logger.info(" load %s into initialized embeddings"%emb_type)
        elif emb_type=='glove':
            pre_trained_emb = vocab.GloVe(name='6B', dim=100)
            logger.info(" load %s into initialized embeddings"%emb_type)
        
        counter = 0.
        for word, index in self.word2index.iteritems():
            try:
                self.embeddings[index] = pre_trained_emb.vectors[pre_trained_emb.stoi[word]]
                counter += 1
            except KeyError:
                pass
            
        logger.info('%d/%d word vectors re-initialized with pre-trained (hit rate: %.2f%%)' % (counter, len(self.word2index), 100*counter/len(self.word2index)))
        
        self.embeddings[self.PAD_index] -= self.embeddings[self.PAD_index]

    def indicies_from_sentence(self, sentence, max_sent_len, padding_place):
      
        if padding_place == 'none':

            sent_padded = sentence
        
        elif padding_place == 'pre':
        
            trunc = sentence[-max_sent_len:]
            
            sent_padded= [self.PAD_word] * max_sent_len
            
            if len(trunc)>0:
                
                sent_padded[-len(trunc):] = trunc
                
        elif padding_place == 'post':
        
            trunc = sentence[:max_sent_len]
            
            sent_padded= [self.PAD_word] * max_sent_len
            
            if len(trunc)>0:
                
                sent_padded[:len(trunc)] = trunc
                    
        output = []

        mask = []
        
        num_hit = 0.0
        
        total = 0.0 
        
        unk_hit = 0.0
        
        for word in sent_padded:
            
            if self.is_number(word):
                
                output.append(self.NUM_index)
                
                num_hit += 1
                
            elif word in self.word2index:
                
                output.append(self.word2index[word])
        
            else:
                
                output.append(self.UNK_index)
                
                unk_hit += 1
                
            if word == self.PAD_word:
            
                mask.append(0)
            
            else:
            
                mask.append(1)
                
                total += 1

        assert len(output) == len(mask) 
    
        return output, mask, (unk_hit,num_hit,total)
    
    def is_number(self, token):
        
        return bool(num_regex.match(token))



    