from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./tmp/tensorboard/runs_test_naive")

for n_iter in range(100):
    writer.add_scalar('Loss/train', n_iter, n_iter)
    writer.add_scalar('Loss/test', n_iter*2, n_iter)
    writer.add_scalar('Accuracy/train', n_iter*3, n_iter)
    writer.add_scalar('Accuracy/test', n_iter*4, n_iter)

# Call flush() method to make sure that all pending events have been written to disk.
writer.flush()

#If you do not need the summary writer anymore, call close() method.
writer.close()