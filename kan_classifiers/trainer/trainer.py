import torch
import numpy as np
from kan_classifiers.eval.benchmark import accuracy
from kan_classifiers.utils.averagemeter import AverageMeter

class CifarTrainer:
    def __init__(self):
        self.train_on_gpu = True if torch.cuda.is_available() else False

    def train_loop(self, 
            train_loader, 
            valid_loader,
            model,
            optimizer,
            criterion,
            model_path,
            n_epochs
        ):

        valid_loss_min = np.Inf # track change in validation loss

        for epoch in range(1, n_epochs+1):

            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            ###################
            # train the model #
            ###################
            model.train()
            for data, target in train_loader:
                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)
                
            ######################    
            # validate the model #
            ######################
            model.eval()
            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
            
            # calculate average losses
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = valid_loss/len(valid_loader.dataset)
                
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), model_path)
                valid_loss_min = valid_loss
    
    

    def evaluate_model(self, test_loader, model, criterion, batch_size):
        # track test loss
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()
        # iterate over test data
        for data, target in test_loader:
            # move tensors to GPU if CUDA is available
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Top 1 and Top 5 accuracy
            # prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            # top1.update(prec1.item(), data.size(0))
            # top5.update(prec5.item(), data.size(0))
            # calculate the batch loss
            loss = criterion(output, target)
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)    
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))

            correct = np.squeeze(correct_tensor.numpy()) if not self.train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # average test loss
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))
        # print('Top 1 accuracy: {:.6f}\n'.format(top1.avg))
        # print('Top 5 accuracy: {:.6f}\n'.format(top5.avg))

        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))