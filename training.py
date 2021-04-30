import torch.nn.functional as F


def train_model(dataset, model, optimizer, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        # training mode
        dataset.set_partition(dataset.train)
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        count = 0
        for x, y in dataset.get_batches():
            # for every batch in the training dataset perform one update step of the optimizer.
            model.zero_grad()
            y_h = model(x)
            loss = F.cross_entropy(y_h, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_correct += (y_h.argmax(-1) == y).float().mean()
            count += 1
        average_train_loss = total_train_loss / count
        average_train_accuracy = total_train_correct / count

        # validation mode
        dataset.set_partition(dataset.valid)
        model.eval()
        total_valid_loss = 0
        total_valid_correct = 0
        count = 0
        for x, y in dataset.get_batches():
            y_h = model(x)
            loss = F.cross_entropy(y_h, y)
            total_valid_loss += loss.item()
            total_valid_correct += (y_h.argmax(-1) == y).float().mean()
            count += 1
        average_valid_loss = total_valid_loss / count
        losses.append((average_train_loss, average_valid_loss))
        average_valid_accuracy = total_valid_correct / count

        print(f'epoch {epoch} accuracies: \t train: {average_train_accuracy}\t valid: {average_valid_accuracy}')
        dataset.shuffle()

    # test mode
    dataset.set_partition(dataset.test)
    model.eval()
    total_test_correct = 0
    count = 0
    for x, y in dataset.get_batches():
        y_h = model(x)
        total_test_correct += (y_h.argmax(-1) == y).float().mean()
        count += 1
    average_test_accuracy = total_test_correct / count

    return losses, (average_train_accuracy, average_valid_accuracy, average_test_accuracy)