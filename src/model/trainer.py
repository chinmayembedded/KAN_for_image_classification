def trainNet(model, lr, trainer, validater):
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)


    # Number of epochs to train for
    loss_keeper={'train':[],'valid':[]}
    acc_keeper={'train':[],'valid':[]}
    train_class_correct = list(0. for i in range(10))
    valid_class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    epochs=50

    # minimum validation loss ----- set initial minimum to infinity
    valid_loss_min = np.Inf 

    for epoch in range(epochs):
        train_loss=0.0
        valid_loss=0.0
        train_correct=0.0
        valid_correct=0.0
        """
        TRAINING PHASE
        """
        model.train() # TURN ON DROPOUT for training
        for images,labels in trainer:
            if use_cuda and torch.cuda.is_available():
                images,labels=images.cuda(),labels.cuda()
            optimizer.zero_grad()
            output=model(images)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            _, pred = torch.max(output, 1)
            train_correct=np.squeeze(pred.eq(labels.data.view_as(pred)))
            for idx in range(batch_size):
                label = labels[idx]
                train_class_correct[label] += train_correct[idx].item()
                class_total[label] += 1

        """
        VALIDATION PHASE
        """
        model.eval() # TURN OFF DROPOUT for validation
        for images,labels in validater:
            if use_cuda and torch.cuda.is_available():
                images,labels=images.cuda(),labels.cuda()
            output=model(images)
            loss=criterion(output,labels)
            valid_loss+=loss.item()
            _, pred = torch.max(output, 1)
            valid_correct=np.squeeze(pred.eq(labels.data.view_as(pred)))
            for idx in range(batch_size):
                label = labels[idx]
                valid_class_correct[label] += valid_correct[idx].item()
                class_total[label] += 1

        # Calculating loss over entire batch size for every epoch
        train_loss = train_loss/len(trainer)
        valid_loss = valid_loss/len(validater)

        # Calculating loss over entire batch size for every epoch
        train_acc=float(100. * np.sum(train_class_correct) / np.sum(class_total))
        valid_acc=float(100. * np.sum(valid_class_correct) / np.sum(class_total))

        # saving loss values
        loss_keeper['train'].append(train_loss)
        loss_keeper['valid'].append(valid_loss)

        # saving acc values
        acc_keeper['train'].append(train_acc)
        acc_keeper['valid'].append(valid_acc)

        print(f"Epoch : {epoch+1}")
        print(f"Training Loss : {train_loss}\tValidation Loss : {valid_loss}")

        if valid_loss<=valid_loss_min:
            print(f"Validation loss decreased from : {valid_loss_min} ----> {valid_loss} ----> Saving Model.......")
            z=type(model).__name__
            torch.save(model.state_dict(), z+'_model.pth')
            valid_loss_min=valid_loss

        print(f"Training Accuracy : {train_acc}\tValidation Accuracy : {valid_acc}\n\n")

    return(loss_keeper,acc_keeper)