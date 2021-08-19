# coding:utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import src.utils as utils
import src.dataset as dataset
import time
from src.utils import alphabet
from src.utils import weights_init

import models.spatial_attention as crnn
print(crnn.__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--path',  default='data/', help='where to ascess image')
parser.add_argument('--trainlist',  default='data/train')
parser.add_argument('--vallist',  default='data/test')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imgH', type=int, default=299, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=299, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=False)
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--model', type=str, default='', help="path to model (to continue training)")
parser.add_argument('--experiment', default='new_expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--adam', default=True, action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
opt = parser.parse_args()
print(opt)

SOS_token = 0
EOS_TOKEN = 1              
BLANK = 2                  


if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir -p {0}'.format(opt.experiment))       

opt.manualSeed = random.randint(1, 10000)  
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = None
train_dataset = dataset.listDataset(path = opt.path, list_file =opt.trainlist, transform=transform)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_dataset = dataset.listDataset(path = opt.path, list_file =opt.vallist, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
test_train = dataset.listDataset(path = opt.path, list_file =opt.trainlist, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
nclass = len(alphabet) + 3         
nc = 3
converter = utils.strLabelConverterForAttention(alphabet)
criterion = torch.nn.NLLLoss()             

model = crnn.Model(opt.nh, nclass, opt.imgW, opt.imgH)
model.apply(weights_init)
if opt.model:
    print('loading pretrained encoder model from %s' % opt.model)
    model.load_state_dict(torch.load(opt.model))

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    model = model.cuda()
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    model_optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.adadelta:
    model_optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
else:
    model_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)
    
def val(model, criterion, batchsize, dataset, teach_forcing=False, max_iter=100, num = 10):
    print('Start')

    for i in model.parameters():
        i.requires_grad = False
    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batchsize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        b = cpu_images.size(0)
        utils.loadData(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1                    

        decoded_words = []
        decoded_label = []
        decoder_attentions = torch.zeros(len(cpu_texts[0]) + 1, opt.max_width)
        target_variable = target_variable.cuda()
        decoder_input = target_variable[0].cuda()  
        decoder_hidden = model.initHidden(b).cuda()
        loss = 0.0
        if not teach_forcing:
            for di in range(1, target_variable.shape[0]):  
                decoder_output, decoder_hidden, decoder_attention = model(decoder_input, decoder_hidden, image)
                loss += criterion(decoder_output, target_variable[di])  
                loss_avg.add(loss)
                decoder_attentions[di-1] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze(1)    
                decoder_input = ni
                if ni == EOS_TOKEN:
                    decoded_words.append('<EOS>')
                    decoded_label.append(EOS_TOKEN)
                    print('smt')
                    break
                else:
                    decoded_words.append(converter.decode(ni))
                    decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1
        if i%num == 0:
            texts = cpu_texts[0]
            print('pred:%-20s, gt: %-20s' % (decoded_words, texts))

    accuracy = n_correct / float(n_total)
    print('Loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(model, criterion, model_optimizer, teach_forcing_prob=1):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    b = cpu_images.size(0)
    target_variable = converter.encode(cpu_texts)
    utils.loadData(image, cpu_images)  
    target_variable = target_variable.cuda()
    decoder_input = target_variable[0].cuda()      
    decoder_hidden = model.initHidden(b).cuda()
    loss = 0.0
    teach_forcing = True if random.random() > teach_forcing_prob else False
    if teach_forcing:
        for di in range(1, target_variable.shape[0]):         
            decoder_output, decoder_hidden, decoder_attention = model(decoder_input, decoder_hidden, image)
            loss += criterion(decoder_output, target_variable[di])         
            decoder_input = target_variable[di] 
    else:
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = model(decoder_input, decoder_hidden, image)
            loss += criterion(decoder_output, target_variable[di])  
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze()
            decoder_input = ni
    model.zero_grad()
    loss.backward()
    model_optimizer.step()
    return loss

if __name__ == '__main__':
    t0 = time.time()
    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader)-1:
            for j in model.parameters():
                j.requires_grad = True
            model.train()
            cost = trainBatch(model, criterion, model_optimizer, teach_forcing_prob=opt.teaching_forcing_prob)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                    (epoch, opt.niter, i, len(train_loader), loss_avg.val()), end=' ')
                loss_avg.reset()
                t1 = time.time()
                print('time elapsed %d' % (t1-t0))
                t0 = time.time()

        # do checkpointing
        if epoch % opt.saveInterval == 0:
            print("For train\n")
            val(model, criterion, 1, dataset=test_train, teach_forcing=False, num = 20)
            print("For test\n")
            val(model, criterion, 1, dataset=test_dataset, teach_forcing=False, num = 20)            # batchsize:1
            torch.save(
                model.state_dict(), '{0}/encoder_{1}.pth'.format(opt.experiment, epoch))