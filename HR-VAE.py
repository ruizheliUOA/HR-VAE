import argparse
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import os
import time
from BatchIter import BatchIter
import numpy as np
from Corpus import Corpus


base_path = "."
print("base_path=", base_path)

parser = argparse.ArgumentParser(description='HR-VAE for PTB or E2E')
parser.add_argument('-bz', '--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('-em', '--embedding-size', type=int, default=512,
                    help='embedding size for training (default: 512)')
parser.add_argument('-hn', '--hidden-size', type=int, default=256,
                    help='hidden size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--no-cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-dt','--dataset', type=str, default="PTB",
                    help='Dataset name')
parser.add_argument('-mw','--min_word_count',type=int, default=1,
                    help='minimum word count')
parser.add_argument('-st','--setting',type=str, default='inputless',
                    help='standard setting or inputless setting')
args = parser.parse_args()

print(args)

if args.dataset == "E2E":
    Train = Corpus(base_path+'/VAE_data/E2E_train.txt', min_word_count=args.min_word_count)
    Eval = Corpus(base_path+'/VAE_data/E2E_valid.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/VAE_data/E2E_test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
elif args.dataset == "PTB":
    Train = Corpus(base_path+'/VAE_data/ptb_train.txt', min_word_count=args.min_word_count)
    Eval = Corpus(base_path+'/VAE_data/ptb_valid.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/VAE_data/ptb_test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)

model_dir = base_path+'/'+args.dataset+'_'+args.setting+'_'+'bz'+str(args.batch_size)+'_'+'hn'+str(args.hidden_size)+'_'+'em'+str(args.embedding_size)+'_model_save/'
recon_dir = base_path+'/'+args.dataset+'_'+args.setting+'_'+'bz'+str(args.batch_size)+'_'+'hn'+str(args.hidden_size)+'_'+'em'+str(args.embedding_size)+'_recon_save/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(recon_dir):
    os.makedirs(recon_dir)


voca_dim = Train.voca_size

print(f"voca_dim={voca_dim}")


emb_dim = args.embedding_size
hid_dim = args.hidden_size
batch_size = args.batch_size


SEED = 999
lr = 0.0001

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available()
                      and not args.no_cuda else 'cpu')

if torch.cuda.is_available():
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print('GPU: %s' % current_gpu_name)

dataloader_train = BatchIter(Train, batch_size)
dataloader_valid = BatchIter(Eval, batch_size)
dataloader_test = BatchIter(Test, batch_size)





class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.layer_dim = (n_layers*2 if bidirectional else n_layers)*hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(dropout)
        
        self.linear_mu = nn.Linear(self.layer_dim*2, self.layer_dim*2)
        self.linear_var = nn.Linear(self.layer_dim*2, self.layer_dim*2)
        

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def get_sen_len(self, sens):
        length = torch.sum(sens > 0, dim=0)
        return length.to(dtype=torch.float)

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sent len, batch size, emb dim]
        mu_ = []
        logvar_ = []
        hx = None
        for i in range(embedded.shape[0]):
            
            _, hx = self.rnn(embedded[i].unsqueeze(0), hx)
            h = self.ziphidden(*hx)
            # cat hidden and cell at each time stamp
            mu = self.linear_mu(h)
            logvar = self.linear_var(h)
            h = self.reparameterize(mu, logvar)  # z = h
            mu_.append(mu)
            logvar_.append(logvar)
        # outputs = [src sent len, batch size, hid dim * n directions]
        # outputs are always from the top hidden layer

        
        mu = torch.stack(mu_)
        logvar = torch.stack(logvar_)
        

        return h, mu, logvar, self.get_sen_len(src)

    def ziphidden(self, hidden, cell):
        b_size = hidden.shape[1]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        h = torch.cat([hidden, cell], dim=2)
        # h = [n layers * n directions, batch size, hid dim * 2]
        h = torch.transpose(h, 0, 1).contiguous()
        # h = [batch size, n layers * n directions, hid dim * 2]
        
            
        
        h = h.view(b_size, -1)
        # h = [batch size, n layers * n directions * hid dim * 2]
        return h

    def loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / mu.shape[0]
        
        return KLD


class Generator2(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout=0.5, bidirectional=False, teacher_force=0.5):
        super(Generator2, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.layer_dim = n_layers*2 if bidirectional else n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        
        self.rnn = nn.LSTM(emb_dim+hid_dim*2, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.teacher_force = teacher_force

    def singledecode(self, input, hidden, cell, lat_z=None):
        # first input to the decoder is the <sos> tokens
        input = input.unsqueeze(0)
        if lat_z.type() != None:
            lat_z = lat_z.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        
        emba_cat = torch.cat([embedded,lat_z], dim=2)

        
        output, (hidden, cell) = self.rnn(emba_cat, (hidden, cell))
        
        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

    def forward(self, input, sen):
        b_size = input.shape[0]
        zz = input.view(b_size, self.layer_dim, -1)
        # zz = [batch size, n layers * n directions, hid dim * 2]
        zzz = torch.transpose(zz, 0, 1)
        # zzz = [n layers * n directions, batch size, hid dim * 2]
        
        hidden = zzz[:, :, :self.hid_dim].contiguous()
        cell = zzz[:, :, self.hid_dim:].contiguous()
        # cell = [n layers * n directions, batch size, hid dim]
        
        # hidden = [n layers * n directions, batch size, hid dim]
        
        max_len = sen.shape[0] #if self.training else sen

        outputs = []
        input = torch.tensor([2]*b_size, device=device)
        for t in range(1, max_len):
            if args.setting in ['standard','inputless']:

                
                output, hidden, cell = self.singledecode(input, hidden, cell, lat_z=zzz[-1,:,:])
                

            outputs.append(output)
            teacher_force = random.random() < self.teacher_force
            top1 = output.max(1)[1]
            if args.setting == 'standard':
                input = sen[t]
            elif args.setting == 'inputless':
                input = torch.tensor([1]*b_size, device=device)
            else:
                input = (sen[t] if self.training and teacher_force else top1)
        output = torch.stack(outputs)
        return output

    def loss(self, prod, target, weight):
        # prod = torch.softmax(prod, 2)
        recon_loss = F.cross_entropy(
            prod.view(-1, prod.shape[2]), target[1:].view(-1),
            ignore_index=0, reduction="sum")
        return recon_loss


encoder = Encoder(voca_dim, emb_dim, hid_dim, 2, 0.5).to(device)
decoder = Generator2(voca_dim, emb_dim, hid_dim, 2).to(device)
opt = optim.Adam(list(encoder.parameters()) +
                 list(decoder.parameters()), lr=lr, eps=1e-6, weight_decay=1e-5)

print(encoder)
print(decoder)

def sentence_acc(prod, target):
    target = target[1:]
    mask = target == 0
    prod = prod.argmax(dim=2)
    prod[mask] = -1
    correct = torch.eq(prod, target).to(dtype=torch.float).sum()
    return correct.item()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(corpus, ep):
    print("--------------------------")
    start_time = time.time()
    encoder.train()
    decoder.train()
    total = 0
    recon_loss_total = 0
    vae_loss_total = 0
    mini_loss_total = 0
    correct_total = 0
    words_total = 0
    batch_total = 0


    
    
    for i, sen in enumerate(corpus):
        # sen: [len_sen, batch]
        batch_size = sen.shape[1]
        opt.zero_grad()
        total += sen.shape[1]
        sen = sen.to(device)
        z, mu, logvar, sen_len = encoder(sen)


        
        
        prod = decoder(z, sen)
        vae_loss = encoder.loss(mu, logvar)
        recon_loss = decoder.loss(prod, sen, sen_len)
        
       
        ((vae_loss+recon_loss)*1).backward()
        opt.step()

        recon_loss_total = recon_loss_total + recon_loss.item()
        vae_loss_total = vae_loss_total + vae_loss.item()
        
        correct = sentence_acc(prod, sen)
        words = sen_len.sum().item()
        correct_total = correct_total + correct
        words_total = words_total + words
        batch_total += batch_size
        

    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(
        f"Time: {epoch_mins}m {epoch_secs}s| Train {ep}: recon_loss={(recon_loss_total/(batch_total)):.04f}, kl_loss={(vae_loss_total/(batch_total)):.04f}, nll_loss={((recon_loss_total+vae_loss_total)/(batch_total)):.04f}, ppl={(math.exp((recon_loss_total+vae_loss_total)/words_total)):.04f}, acc={(correct_total/words_total):.04f}")
    return recon_loss_total/(batch_total), vae_loss_total/(batch_total), correct_total/words_total, (recon_loss_total+vae_loss_total)/(batch_total), math.exp((recon_loss_total+vae_loss_total)/words_total)

# =================================


def get_sentence(batch):
    sens = []
    for b in range(batch.shape[1]):
        sen = [Train.id_word[batch[i, b].item()]
               for i in range(batch.shape[0])]
        sens.append(" ".join(sen))
    return sens


def get_label(batch):
    labs = []
    for b in range(batch.shape[0]):
        index = (batch[b] > 0.99).nonzero().view(-1).tolist()
        lab = [Train.id_class[i] for i in index]
        labs.append(str(lab))
    return labs


def reconstruction(corpus, ep):
    
    encoder.eval()
    
    decoder.eval()
    out_org = []
    
    out_recon_mu = []
    
    recon_loss_total = 0
    vae_loss_total = 0
    words_total = 0
    batch_total = 0

    
    
    
    for i, sen in enumerate(corpus):
        b_size = sen.shape[1]
        out_org += get_sentence(sen[1:])
        sen = sen.to(device)
        #
        with torch.no_grad():
            z, mu, logvar, sen_len= encoder(sen)
            
            recon_mu = decoder(z, sen)
            vae_loss = encoder.loss(mu, logvar)
            recon_loss = decoder.loss(recon_mu, sen, sen_len)
            # recon = [sen len, batch size, voca size]
            
            sens_mu = recon_mu.argmax(dim=2)
            
            out_recon_mu += get_sentence(sens_mu.to("cpu"))

            recon_loss_total = recon_loss_total + recon_loss.item()
            vae_loss_total = vae_loss_total + vae_loss.item()
            
            words = sen_len.sum().item()
            
            words_total = words_total + words
            batch_total += b_size
    print(f"Eval: words_total:{words_total}, batch_total:{batch_total}, recon_loss:{(recon_loss_total/(batch_total)):.04f}, kl_loss:{(vae_loss_total/(batch_total)):.04f}, nll_loss:{((recon_loss_total+vae_loss_total)/(batch_total)):.04f}, ppl:{(math.exp((recon_loss_total+vae_loss_total)/words_total)):.04f}")

    
    text = []
    for i in range(len(out_recon_mu)):
        text.append("origion: " + out_org[i])
        
        text.append("reco_mu: " + out_recon_mu[i])
        
        text.append("\n")
    with open(recon_dir+f"LstmVae_outcome_lx_{ep}.txt", "w") as f:
        f.write("\n".join(text))

    
            

    return math.exp((recon_loss_total+vae_loss_total)/words_total)





ep = 0
# ============== run ==============
if args.load:
    state = torch.load(model_dir+'LstmVae.tch_lx')
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    state2 = torch.load(model_dir+'LstmVae.tchopt_lx')
    ep = state2["ep"]+1
    opt.load_state_dict(state2["opt"])


history = []
Best_ppl = 1e5
for ep in range(ep, args.epochs):
    recon_loss, var_loss, acc, nll_loss, ppl = train(dataloader_train, ep)
    history.append(f"{ep}\t{recon_loss}\t{var_loss}\t{acc}\t{nll_loss}\t{ppl}")
    with open(model_dir+'LstmVae_loss_lx.txt', 'w') as f:
        f.write("\n".join(history))
    
    eval_ppl = reconstruction(dataloader_valid, ep)
    test_ppl = reconstruction(dataloader_test, ep)
    
    if args.save and eval_ppl < Best_ppl:
        Best_ppl = eval_ppl
        
        state = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
        }
        torch.save(state, model_dir + 'LstmVae.tch_lx')
        state2 = {
            "opt": opt.state_dict(),
            "ep": ep
        }
        torch.save(state2, model_dir + 'LstmVae.tchopt_lx')
    
