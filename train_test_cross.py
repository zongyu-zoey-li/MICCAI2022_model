from __future__ import division
from __future__ import print_function

import os
import numpy as np
from ray.tune.session import checkpoint_dir
import torch 
import torch.nn as nn
from random import randrange
from data_loading import RawFeatureDataset
from lstm_model import LSTM_Layer
import pandas as pd
from logger import Logger
import utils
import pdb

from config import (raw_feature_dir, sample_rate,
                    gesture_class_num, dataset_name)


# for parameter tuning LSTM

from config import input_size, num_class, raw_feature_dir, validation_trial, validation_trial_train, tcn_model_params 
from utils import get_cross_val_splits
import ray
from ray import tune
from tcn_model import EncoderDecoderNet

config = { "hidden_size":tune.sample_from(lambda _: 2**np.random.randint(3,9)), "num_layers":tune.choice([1,2,3,4]),
    "learning_rate":tune.loguniform(1e-5,1e-2),
        "batch_size":1, "weight_decay": tune.loguniform(1e-3,1e-1)}
def train_model_parameter( config, type,input_size, num_class,num_epochs,dataset_name,sample_rate,
                loss_weights=None, 
                trained_model_file=None, 
                log_dir=None, checkpoint_dir=None):

    if type =='lstm':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTM_Layer(input_size=input_size, num_class=num_class,hidden_size=config["hidden_size"], num_layers=config["num_layers"],device=device)
        model.to(device)
    if type =='tcn':
        model = EncoderDecoderNet(**tcn_model_params['model_params'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    paths = get_cross_val_splits(validation = True)

    train_trail_list = paths["train"]
    test_trail_list = paths["test"]
    train_dataset = RawFeatureDataset(dataset_name, 
                                        train_trail_list,
                                        feature_type="sensor",
                                        sample_rate=sample_rate,
                                        sample_aug=False,
                                        normalization=[None, None])
    #breakpoint()

    test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
    val_dataset = RawFeatureDataset(dataset_name, 
                                        test_trail_list,
                                        feature_type="sensor",
                                        sample_rate=sample_rate,
                                        sample_aug=False,
                                        normalization=test_norm)

    loss_weights = utils.get_class_weights(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=config["batch_size"], shuffle=True)
   
    
    model.train()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).to(device), #.cuda()
                        ignore_index=-1)

    # Logger
    if log_dir is not None:
        logger = Logger(log_dir) 

    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],
                                            weight_decay=config["weight_decay"])
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    step = 1
    for epoch in range(num_epochs):
        print(epoch)
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):

            feature = data['feature'].float()
            feature = feature.to(device) #.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.to(device) #.cuda()
            #print(feature.shape)
            #print(gesture.shape)

            # Forward
            out = model(feature)
           # print(out.shape)
            flatten_out = out.view(-1, out.shape[-1])
           # print(flatten_out.shape)

            #breakpoint()

            loss = criterion(input=flatten_out, target=gesture)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 10 == 9:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

            # Logging
            if log_dir is not None:
                logger.scalar_summary('loss', loss.item(), step)

            step += 1

        train_result = test_model(model, train_dataset, loss_weights)
        t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

        val_result = test_model(model, val_dataset, loss_weights)
        v_accuracy, v_edit_score, v_loss, v_f_scores = val_result

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            if epoch ==num_epochs -1:
                path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(epoch))
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=v_loss , accuracy=v_accuracy,edit_score=v_edit_score,F1=v_f_scores)
        print("Finished Training")
        if log_dir is not None:
            train_result = test_model(model, train_dataset, loss_weights)
            t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

            val_result = test_model(model, val_dataset, loss_weights)
            v_accuracy, v_edit_score, v_loss, v_f_scores = val_result

            logger.scalar_summary('t_accuracy', t_accuracy, epoch)
            logger.scalar_summary('t_edit_score', t_edit_score, epoch)
            logger.scalar_summary('t_loss', t_loss, epoch)
            logger.scalar_summary('t_f_scores_10', t_f_scores[0], epoch)
            logger.scalar_summary('t_f_scores_25', t_f_scores[1], epoch)
            logger.scalar_summary('t_f_scores_50', t_f_scores[2], epoch)
            logger.scalar_summary('t_f_scores_75', t_f_scores[3], epoch)

            logger.scalar_summary('v_accuracy', v_accuracy, epoch)
            logger.scalar_summary('v_edit_score', v_edit_score, epoch)
            logger.scalar_summary('v_loss', v_loss, epoch)
            logger.scalar_summary('v_f_scores_10', v_f_scores[0], epoch)
            logger.scalar_summary('v_f_scores_25', v_f_scores[1], epoch)
            logger.scalar_summary('v_f_scores_50', v_f_scores[2], epoch)
            logger.scalar_summary('v_f_scores_75', v_f_scores[3], epoch)

        if trained_model_file is not None:
            torch.save(model.state_dict(), trained_model_file)


def train_model(config,type,train_dataset,val_dataset,input_size, num_class,num_epochs,
                loss_weights=None, 
                trained_model_file=None, 
                log_dir=None, checkpoint_dir=None,model_index=0):

    if type =='lstm':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTM_Layer(input_size=input_size, num_class=num_class,hidden_size=config["hidden_size"], num_layers=config["num_layers"],device=device)
        model.to(device)
    if type =='tcn':
        model = EncoderDecoderNet(**tcn_model_params['model_params'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
    
    if model_index==5:
        df1 = pd.DataFrame(index=np.arange(0,num_epochs),  columns=('t_accuracy','t_edit_score','t_loss','t_f_scores_10','t_f_scores_25','t_f_scores_50','t_f_scores_75',\
    'v_accuracy','v_edit_score','v_loss','v_f_scores_10','v_f_scores_25','v_f_scores_50','v_f_scores_75'))
        df2 = pd.DataFrame(index=np.arange(0,num_epochs),  columns=('t_accuracy','t_edit_score','t_loss','t_f_scores_10','t_f_scores_25','t_f_scores_50','t_f_scores_75',\
    'v_accuracy','v_edit_score','v_loss','v_f_scores_10','v_f_scores_25','v_f_scores_50','v_f_scores_75'))#breakpoint()
        df = [df1, df2]
    else:
        df = pd.DataFrame(index=np.arange(0,num_epochs),  columns=('t_accuracy','t_edit_score','t_loss','t_f_scores_10','t_f_scores_25','t_f_scores_50','t_f_scores_75',\
    'v_accuracy','v_edit_score','v_loss','v_f_scores_10','v_f_scores_25','v_f_scores_50','v_f_scores_75'))
      
       
    loss_weights = utils.get_class_weights(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=config["batch_size"], shuffle=True)
   
    
    model.train()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).to(device), #.cuda()
                        ignore_index=-1)

    

    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],
                                            weight_decay=config["weight_decay"])
    
    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, "checkpoint"))
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    step = 1
    for epoch in range(num_epochs):
        print(epoch)
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):

            feature = data['feature'].float()
            feature = feature.to(device) #.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.to(device) #.cuda()

            # Forward
            out = model(feature)
            flatten_out = out.view(-1, out.shape[-1])
            #breakpoint()

            loss = criterion(input=flatten_out, target=gesture)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 10 == 9:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

            # Logging
            #if log_dir is not None:
                #logger.scalar_summary('loss', loss.item(), step)

            step += 1

        if trained_model_file is not None:
            if not os.path.exists(trained_model_file):
                os.makedirs(trained_model_file, exist_ok=True)
            file_dir = os.path.join(trained_model_file,"checkpoint_{}.pth".format(epoch))
            torch.save(model.state_dict(), file_dir)

        if log_dir is not None:
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            train_result = test_model(model, train_dataset, loss_weights)
            t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

            if model_index==5:
                for table_idx, val in enumerate(val_dataset):
                    val_result = test_model(model, val, loss_weights)
                    v_accuracy, v_edit_score, v_loss, v_f_scores = val_result
                    df[table_idx].loc[epoch] = [t_accuracy, t_edit_score,t_loss, t_f_scores[0], t_f_scores[1], t_f_scores[2], t_f_scores[3],\
                    v_accuracy, v_edit_score,v_loss, v_f_scores[0], v_f_scores[1], v_f_scores[2], v_f_scores[3]]
                df[0].to_csv(os.path.join(log_dir,'train_test_result_JIGSAWS.csv'))
                df[1].to_csv(os.path.join(log_dir,'train_test_result_DESK.csv'))

            else:
                val_result = test_model(model, val_dataset, loss_weights)
                v_accuracy, v_edit_score, v_loss, v_f_scores = val_result
                df.loc[epoch] = [t_accuracy, t_edit_score,t_loss, t_f_scores[0], t_f_scores[1], t_f_scores[2], t_f_scores[3],\
                    v_accuracy, v_edit_score,v_loss, v_f_scores[0], v_f_scores[1], v_f_scores[2], v_f_scores[3]]
                df.to_csv(os.path.join(log_dir,'train_test_result.csv'))
        

def test_model(model, test_dataset, loss_weights=None, plot_naming=None):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()

    if loss_weights is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(
                        weight=torch.Tensor(loss_weights).to(device), #.cuda()
                        ignore_index=-1)

    #Test the Model
    total_loss = 0
    preditions = []
    gts=[]

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            
            feature = data['feature'].float()
            feature = feature.to(device) #.cuda()

            gesture = data['gesture'].long()
            gesture = gesture.view(-1)
            gesture = gesture.to(device) #.cuda()

            # Forward
            out = model(feature)
            out = out.squeeze(0)

            loss = criterion(input=out, target=gesture)

            total_loss += loss.item()

            pred = out.data.max(1)[1]

            trail_len = (gesture.data.cpu().numpy()!=-1).sum()
            gesture = gesture[:trail_len]
            pred = pred[:trail_len]
            
            preditions.append(pred.cpu().numpy())
            gts.append(gesture.data.cpu().numpy())

            # Plot   [Errno 2] No such file or directory: './graph/JIGSAWS/Suturing/sensor_run_1_split_1_seq_0.png' 12/7/21
            # if plot_naming:    
            #     graph_file = os.path.join(graph_dir, '{}_seq_{}'.format(
            #                                     plot_naming, str(i)))

            #     utils.plot_barcode(gt=gesture.data.cpu().numpy(), 
            #                        pred=pred.cpu().numpy(), 
            #                        visited_pos=None,
            #                        show=False, save_file=graph_file)

    bg_class = 0 if dataset_name != 'JIGSAWS' else None

    avg_loss = total_loss / len(test_loader.dataset)
    edit_score = utils.get_edit_score_colin(preditions, gts,
                                            bg_class=bg_class)
    accuracy = utils.get_accuracy_colin(preditions, gts)
    #accuracy = utils.get_accuracy(preditions, gts)
    
    f_scores = []
    for overlap in [0.1, 0.25, 0.5, 0.75]:
        f_scores.append(utils.get_overlap_f1_colin(preditions, gts,
                                        n_classes=gesture_class_num, 
                                        bg_class=bg_class, 
                                        overlap=overlap))

    model.train()
    return accuracy, edit_score, avg_loss, f_scores





######################### Main Process #########################

def cross_validate(dataset_name,net_name,model_index=0):
    '''

    '''
    if net_name =='tcn'and dataset_name=="JIGSAWS":
        num_epochs = 50 # about 25 mins for 5 fold cross validation
        config= {'learning_rate': 0.0007358358290370388, 'batch_size': 1, 'weight_decay': 0.0002967511175393983} # for orientation
       # for velocity config = {'learning_rate': 0.00027347155281573553, 'batch_size': 1, 'weight_decay': 0.00037328332914909917} #EPOCH=30 tcn
    # if net_name=='lstm':
    #     num_epochs = 60
    #     config =  {'hidden_size': 128 , 'learning_rate': 0.000145129 ,  'num_layers': 3 ,'batch_size': 1, 'weight_decay':0.00106176 } # Epoch =60 lstm
    
    if net_name=='tcn' and dataset_name=="DESKpegtransfer":
        num_epochs = 100
        config=  {'learning_rate': 0.0007098462302555751, 'batch_size': 1, 'weight_decay': 0.0007278759551827732}
     
    
    
    # Get trial splits
    cross_val_splits = utils.get_cross_val_splits()
    
    #breakpoint()

    # Cross-Validation Result
    #result = []

    # Cross Validation
    for idx, data in enumerate(cross_val_splits):
        #breakpoint()
        # Dataset
        train_dir, test_dir,name = data['train'], data['test'],data['name']
        import fnmatch
        if model_index==5:
            z=[dir  for dir in test_dir if fnmatch.fnmatch(dir,"*[Suturing,Needle_Passing,Knot_Tying]*")]
            test_dir_5a=z
            z=[dir  for dir in test_dir if fnmatch.fnmatch(dir,"*DESKpegtransfer*")]
            test_dir_5b=z
            test_dir_model5 = [test_dir_5a,test_dir_5b]
            train_dataset = RawFeatureDataset(dataset_name, 
                                            train_dir,
                                            feature_type="sensor",
                                            sample_rate=sample_rate,
                                            sample_aug=False,
                                            normalization=[None, None])
            test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
            test_dataset_5a = RawFeatureDataset(dataset_name, 
                                            test_dir_model5[0],
                                            feature_type="sensor",
                                            sample_rate=sample_rate,
                                            sample_aug=False,
                                            normalization=test_norm)
            test_dataset_5b = RawFeatureDataset(dataset_name, 
                                            test_dir_model5[1],
                                            feature_type="sensor",
                                            sample_rate=sample_rate,
                                            sample_aug=False,
                                            normalization=test_norm)
            loss_weights = utils.get_class_weights(train_dataset)
            # make directories
            path = os.getcwd()
            trained_model_dir=  os.path.join(path,dataset_name,net_name,name) # contain name of the testing set
            os.makedirs(trained_model_dir, exist_ok=True)
            log_dir = os.path.join(trained_model_dir,'log')
            checkpoint_dir = os.path.join(trained_model_dir,'checkpoints')
            test_dataset = [test_dataset_5a,test_dataset_5b]
            train_model(config,net_name,train_dataset,test_dataset,input_size, num_class,num_epochs,
                    loss_weights=loss_weights, 
                    trained_model_file=trained_model_dir, 
                    log_dir=log_dir, checkpoint_dir=checkpoint_dir,model_index=5)
        else:
            train_dataset = RawFeatureDataset(dataset_name, 
                                            train_dir,
                                            feature_type="sensor",
                                            sample_rate=sample_rate,
                                            sample_aug=False,
                                            normalization=[None, None])
            #breakpoint()

            test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
            test_dataset = RawFeatureDataset(dataset_name, 
                                            test_dir,
                                            feature_type="sensor",
                                            sample_rate=sample_rate,
                                            sample_aug=False,
                                            normalization=test_norm)

            loss_weights = utils.get_class_weights(train_dataset)
            # make directories
            path = os.getcwd()
            trained_model_dir=  os.path.join(path,dataset_name,net_name,name) # contain name of the testing set
            os.makedirs(trained_model_dir, exist_ok=True)
            log_dir = os.path.join(trained_model_dir,'log')
            checkpoint_dir = os.path.join(trained_model_dir,'checkpoints')
            
            train_model(config,net_name,train_dataset,test_dataset,input_size, num_class,num_epochs,
                    loss_weights=loss_weights, 
                    trained_model_file=trained_model_dir, 
                    log_dir=log_dir, checkpoint_dir=checkpoint_dir)

        #acc, edit, _, f_scores = test_model(model, test_dataset, 
                                         #   loss_weights=loss_weights)


if __name__ == "__main__":
    cross_validate(dataset_name,'tcn')
