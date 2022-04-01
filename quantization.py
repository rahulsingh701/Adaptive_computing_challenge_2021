import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import random_split,DataLoader
import sys
import argparse
import os

from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from pathlib import Path




DIVIDER = '-----------------------------------------'

def evaluate(model,test_dataset,test_loader,device):
  test_error_count = 0.0
  for images, labels in iter(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
  
  test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
  return test_accuracy
  #print('the test accuracy is = ',test_accuracy)






def quantization(model,build_dir,batch_size,quant_mode):
    val_percent = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dir_checkpoint = Path('./checkpoints/')
    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    dataset = datasets.ImageFolder(
    'Collision_Avoidance',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

  

    net = model
    net.load_state_dict(torch.load(os.path.join(float_model,'best_model_resnet18.pth'),map_location='cuda:0'))
    net.cuda()
    
    if (quant_mode=='test'):
        batch_size = 1
    #[channel,height,width]
    rand_in = torch.randn([batch_size, 3, 224, 224]).cuda()
    quantizer = torch_quantizer(quant_mode, net, (rand_in), output_dir=quant_model) 
    quantized_model = quantizer.quant_model
    
    

    val_score = evaluate(quantized_model,val_set,val_loader,device)
    print('val_score is = ',val_score)

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
    return

def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=1,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()
  model = torchvision.models.resnet18(pretrained=False)
  model.fc = torch.nn.Linear(512, 2)  
  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantization(model,args.build_dir,args.batchsize,args.quant_mode)

  return



if __name__ == '__main__':
    run_main()
