
from config import input_size, num_class, raw_feature_dir, validation_trial, validation_trial_train, sample_rate,dataset_name
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from functools import partial
from lstm_model import LSTM_Layer
from train_test_cross import train_model_parameter, test_model
import numpy as np
import torch
import os
import pdb
from utils import get_cross_val_splits
from data_loading import RawFeatureDataset
from logger import Logger
import utils
import torch 
import torch.nn as nn

def main(num_samples=1, max_num_epochs=50):
    #config = { "hidden_size":tune.sample_from(lambda _: 2**np.random.randint(6,10)), "num_layers":tune.choice([2,3,4,6,8]),
      # "learning_rate":tune.loguniform(1e-5,1e-3),
       #    "batch_size":1, "weight_decay": tune.loguniform(1e-3,1e-2)}
    #config = {'hidden_size': 64, 'num_layers': 2, 'learning_rate': 0.00045955128646685565, 'batch_size': 1, 'weight_decay': 0.0035444727431377843}
    config =  {'hidden_size': 128 , 'learning_rate': 0.000145129 ,  'num_layers': 3 ,'batch_size': 1, 'weight_decay':0.00106176 } # DEFAULT_bc2a4_00002
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_model_parameter,type ='lstm',input_size=input_size,\
             num_class=num_class,num_epochs=max_num_epochs,dataset_name=dataset_name,\
                 sample_rate=sample_rate),
        resources_per_trial={"cpu": 10, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
   

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

def main_tcn(num_samples=1, max_num_epochs=50):


    config = {'learning_rate': 0.0003042861945575232, 'batch_size': 1, 'weight_decay': 0.00012035748692105724} #EPOCH=30
    #config = {"learning_rate":tune.loguniform(1e-5,1e-3), "batch_size":1, "weight_decay": tune.loguniform(1e-4,1e-2)}
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_model_parameter,type='tcn',input_size=input_size,\
             num_class=num_class,num_epochs=max_num_epochs,dataset_name=dataset_name,\
                 sample_rate=sample_rate),
        resources_per_trial={"cpu": 10, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
   

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    
# Best trial config: {'hidden_size': 64, 'num_layers': 2, 'learning_rate': 0.00045955128646685565, 'batch_size': 1, 'weight_decay': 0.0035444727431377843}

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
   main_tcn(num_samples=1, max_num_epochs=60)
    # config = { "hidden_size":2, "num_layers":2,
    #     "learning_rate":1e-5,
    #         "batch_size":1, "weight_decay": 1e-3}
    # train_model_parameter( config,input_size, num_class,num_epochs=1,dataset_name=dataset_name,sample_rate=sample_rate,
    #             loss_weights=None, 
    #             trained_model_file=None, 
    #             log_dir=None, checkpoint_dir=None)
    # num_epochs=1
    # dataset_name="JIGSAWS"
    # loss_weights=None
    # trained_model_file=None
    # log_dir=None
    # checkpoint_dir=None
   
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = LSTM_Layer(input_size=input_size, num_class=num_class,hidden_size=config["hidden_size"], num_layers=config["num_layers"],device=device)
    # model.to(device)
    # paths = get_cross_val_splits(validation = True)

    # train_trail_list = paths["train"]
    # test_trail_list = paths["test"]
    # train_dataset = RawFeatureDataset(dataset_name, 
    #                                     train_trail_list,
    #                                     feature_type="sensor",
    #                                     sample_rate=sample_rate,
    #                                     sample_aug=False,
    #                                     normalization=[None, None])
    # #breakpoint()

    # test_norm = [train_dataset.get_means(), train_dataset.get_stds()]
    # val_dataset = RawFeatureDataset(dataset_name, 
    #                                     test_trail_list,
    #                                     feature_type="sensor",
    #                                     sample_rate=sample_rate,
    #                                     sample_aug=False,
    #                                     normalization=test_norm)

    # loss_weights = utils.get_class_weights(train_dataset)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                 batch_size=config["batch_size"], shuffle=True)

    # #breakpoint
    # model.train()

    # if loss_weights is None:
    #     criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # else:
    #     criterion = nn.CrossEntropyLoss(
    #                     weight=torch.Tensor(loss_weights).to(device), #.cuda()
    #                     ignore_index=-1)

    # # Logger
    # if log_dir is not None:
    #     logger = Logger(log_dir) 

    # optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],
    #                                         weight_decay=config["weight_decay"])
    
    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, "checkpoint"))
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    # step = 1
    # for epoch in range(num_epochs):
    #     print(epoch)
    #     running_loss = 0.0
    #     epoch_steps = 0
    #     for i, data in enumerate(train_loader):

    #         feature = data['feature'].float()
    #         feature = feature.to(device) #.cuda()

    #         gesture = data['gesture'].long()
    #         gesture = gesture.view(-1)
    #         gesture = gesture.to(device) #.cuda()
            
    #         # Forward
    #         out = model(feature)
    #         #breakpoint()
    #         flatten_out = out.view(-1, out.shape[-1])
    #         #breakpoint()

    #         loss = criterion(input=flatten_out, target=gesture)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         epoch_steps += 1
    #         if i % 10 == 9:  # print every 2000 mini-batches
    #             print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
    #                                             running_loss / epoch_steps))
    #             running_loss = 0.0

    #         # Logging
    #         if log_dir is not None:
    #             logger.scalar_summary('loss', loss.item(), step)

    #         step += 1

    #     train_result = test_model(model, train_dataset, loss_weights)
    #     t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

    #     val_result = test_model(model, val_dataset, loss_weights)
    #     v_accuracy, v_edit_score, v_loss, v_f_scores = val_result

    #     with tune.checkpoint_dir(epoch) as checkpoint_dir:
    #         path = os.path.join(checkpoint_dir, "checkpoint")
    #         torch.save((model.state_dict(), optimizer.state_dict()), path)

    #     tune.report(loss=v_loss , accuracy=v_accuracy)
    #     print("Finished Training")
    #     if log_dir is not None:
    #         train_result = test_model(model, train_dataset, loss_weights)
    #         t_accuracy, t_edit_score, t_loss, t_f_scores = train_result

    #         val_result = test_model(model, val_dataset, loss_weights)
    #         v_accuracy, v_edit_score, v_loss, v_f_scores = val_result

    #         logger.scalar_summary('t_accuracy', t_accuracy, epoch)
    #         logger.scalar_summary('t_edit_score', t_edit_score, epoch)
    #         logger.scalar_summary('t_loss', t_loss, epoch)
    #         logger.scalar_summary('t_f_scores_10', t_f_scores[0], epoch)
    #         logger.scalar_summary('t_f_scores_25', t_f_scores[1], epoch)
    #         logger.scalar_summary('t_f_scores_50', t_f_scores[2], epoch)
    #         logger.scalar_summary('t_f_scores_75', t_f_scores[3], epoch)

    #         logger.scalar_summary('v_accuracy', v_accuracy, epoch)
    #         logger.scalar_summary('v_edit_score', v_edit_score, epoch)
    #         logger.scalar_summary('v_loss', v_loss, epoch)
    #         logger.scalar_summary('v_f_scores_10', v_f_scores[0], epoch)
    #         logger.scalar_summary('v_f_scores_25', v_f_scores[1], epoch)
    #         logger.scalar_summary('v_f_scores_50', v_f_scores[2], epoch)
    #         logger.scalar_summary('v_f_scores_75', v_f_scores[3], epoch)

    #     if trained_model_file is not None:
    #         torch.save(model.state_dict(), trained_model_file)





