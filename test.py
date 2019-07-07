import torch
import my_model
import progressbar
import torch.nn as nn
import my_dataloader

batch_size = my_dataloader.batch_size
model = my_model.Model().cuda().half()
model.load_state_dict(torch.load('../data/history/2019-07-06 15:14:26.869166/11:8500.model'))
model.train()
bar = progressbar.ProgressBar(maxval=len(my_dataloader.test_loader.dataset)/batch_size, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

def test(model):
    with torch.no_grad():
        bar.start()
        model.train()
        correct_count = 0 
        sum_count = 0
        for j, (input_batch, label_batch) in enumerate(my_dataloader.test_loader):
            input_batch = input_batch.cuda()
            input_batch = input_batch.half()
            input_batch = input_batch / 255
            label_batch = label_batch.cuda().squeeze()
            loss_1, outputs = model(input_batch)
            _, predicted = torch.max(outputs,1)
            sum_count += batch_size
            correct_count += (predicted == label_batch).sum().item()
            bar.update(j)
        bar.finish()
        return correct_count / sum_count * 100

test_accuracy = test(model)
print("test_accuracy: %.2f"%test_accuracy)

