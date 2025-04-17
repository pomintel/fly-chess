from tqdm import tqdm
import torch

def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss, total = 0.0, 0,
    pbar = tqdm(train_loader,unit="batch", desc="Training")
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        output = model(data)
        if getattr(model, "pruning", False):
            ce_loss = criterion(output, target)
            l1_loss = model.lambda_l1 * model.get_l1_loss() if model.lambda_l1 is not None else 0
            loss = ce_loss + l1_loss
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        try:
            batch_size = data.size(0)
        except:
            batch_size = data[0].size(0)
        total_loss += loss.item() * batch_size
        total += batch_size


    return total_loss / total

def train_steps(model, optimizer, criterion, train_loader, data_iter, device,interval_steps,step,epoch):
    model.train()
    total_loss, total = 0.0, 0
    with tqdm(total=interval_steps, desc=f"Training Steps {step} to {step + interval_steps - 1}") as pbar:
        for _ in range(interval_steps):
            try:
                data, target = next(data_iter)
            except StopIteration:
                epoch += 1
                print(f"Starting epoch {epoch}")
                data_iter = iter(train_loader)
                data, target = next(data_iter)

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            if getattr(model, "pruning", False):
                ce_loss = criterion(output, target)
                l1_loss = model.lambda_l1 * model.get_l1_loss() if model.lambda_l1 is not None else 0
                loss = ce_loss + l1_loss
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            try:
                batch_size = data.size(0)
            except:
                batch_size = data[0].size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            step += 1
            pbar.update(1)
    return total_loss / total, data_iter, step, epoch


def eval_epoch(model, criterion, test_loader, device):
    model.eval()

    with torch.no_grad():

        total_loss, total = 0.0, 0,
        pbar = tqdm(test_loader,unit="batch", desc="Eval", leave=True)
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            if getattr(model, "pruning", False):
                ce_loss = criterion(output, target)
                l1_loss = model.lambda_l1 * model.get_l1_loss() if model.lambda_l1 is not None else 0
                loss = ce_loss + l1_loss
            else:
                loss = criterion(output, target)
            try:
                batch_size = data.size(0)
            except:
                batch_size = data[0].size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

        return total_loss / total

