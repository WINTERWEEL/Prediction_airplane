import os
import sys
# sys.path.append('C:/Users/WINTER/Desktop/Python基础/report')
import torch.utils.data
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from EarlyStopping import EarlyStopping
from dataset import AirlineData
from args import args_parser
from model import MultiLayer

def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args):
    """
    train and test the model
    :param args: config dict
    :return: no return, generate a logfile and checkpoint file
    """
    set_seeds(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset and dataloader
    train_dataset = AirlineData(tag='train')
    valid_dataset = AirlineData(tag='valid')
    test_dataset = AirlineData(tag='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True)

    # model
    args.num_feature = train_dataset.num_features
    model = MultiLayer(args)
    filename = ('epoch_'+str(args.epoch)+'_lr_'+str(args.lr)+'_batch_'+str(args.batch)+"_num_layers_"
                +str(args.num_layers)+"_hidden_size_"+str(args.hidden_size) +"_optimizer_"+args.optimizer
                +"_dropout_"+str(args.dropout)+"_batchnorm_"+str(args.batch_norm) )
    args.name = filename

    # optimizer
    if args.optimizer== 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, nesterov=True)
    elif args.optimizer== 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,weight_decay=1e-5)
    else:
        raise ValueError("Optimizer can not be identified")
    loss_func=torch.nn.MSELoss(reduction='mean')

    # log
    log_path = './logs/'+args.name+'.log'
    logfile = open(log_path, "a")
    logfile.write("epoch "+str(args.epoch)+"\n")
    logfile.write("lr "+str(args.lr)+"\n")
    logfile.write("batch "+str(args.batch)+"\n")
    logfile.write("num_layers "+str(args.num_layers)+"\n")
    logfile.write("hidden_size "+str(args.hidden_size)+"\n")
    logfile.write("optimizer "+args.optimizer+"\n")
    logfile.write("dropout "+str(args.dropout)+"\n")
    logfile.write("batchnorm "+str(args.batch_norm)+"\n")

    # checkpoint
    model_save_path = './models/' + args.name
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_model_path = model_save_path+'/best_model.pth'
    early_stopping = EarlyStopping(best_model_path, args.patience, verbose=True)
    # train
    for epoch in tqdm(range(args.epoch)):
        train_loss_epoch = 0
        idx = 0
        for x, y in tqdm(train_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            idx += 1
            output = model(x)
            loss = loss_func(output, torch.unsqueeze(y, -1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()

        # valid
        valid_loss_epoch = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(valid_loader):
                x = x.float().to(device)
                y = y.float().to(device)
                pred = model(x)
                valid_loss_epoch += loss_func(pred, torch.unsqueeze(y, -1)).item()
        valid_loss_epoch = valid_loss_epoch / (idx + 1)
        logfile.write("epoch{} : Train Loss: {:.4f}  Valid Loss {:.4f}\n".format(epoch+1,train_loss_epoch / idx,valid_loss_epoch))
        print("epoch{} : Train Loss: {:.4f}  Valid Loss {:.4f}\n".format(epoch,train_loss_epoch / idx,valid_loss_epoch))
        torch.save(model.state_dict(), model_save_path + '/' + f'epoch_{epoch + 1}_model.pth')

        # early stop
        early_stopping(valid_loss_epoch, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(best_model_path))

    # test
    test_loss = test(test_loader, model, device, loss_func)
    print("test loss : ", test_loss)
    logfile.write("test loss : "+str(test_loss))


def test(test_loader,model,device,loss_func):
    """
    test the best model after early stopping
    :param test_loader: testset dataloader
    :param model: best model
    :param device: current device
    :param loss_func: loss function
    :return: test loss
    """
    test_loss= 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            test_loss += loss_func(pred, torch.unsqueeze(y, -1)).item()
    test_loss = test_loss / (idx + 1)
    return test_loss

if __name__ == '__main__':
    args = args_parser() # load parameters
    train(args)
