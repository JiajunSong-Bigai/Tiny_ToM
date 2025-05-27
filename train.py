import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os

from generator import ScenarioGenerator
from utils import i_to_c, c_to_i


def prepare_batch(batch_data, device, pad_token=0):
    """
    Prepare a batch of data for training by tokenizing and padding sequences.

    Args:
        batch_data: List of dictionaries containing 'events', 'question', and 'answer'
        device: Device to move tensors to
        pad_token: Token used for padding

    Returns:
        src: Tensor of tokenized and padded source sequences (events + question)
        tgt: Tensor of tokenized and padded target sequences (answer)
        src_mask: Mask for source sequences (1 for actual tokens, 0 for padding)
    """
    # Tokenize events, questions, and answers
    src_sequences = []
    tgt_sequences = []

    for item in batch_data:
        # Tokenize events + question (input)
        events_tokens = item["events"]
        question_tokens = item["question"]
        src_tokens = c_to_i(f"{events_tokens} ; {question_tokens}")
        src_sequences.append(src_tokens)

        # Tokenize answer (target)
        answer_tokens = c_to_i(item["answer"])
        tgt_sequences.append(answer_tokens)

    max_src_len = max(len(seq) for seq in src_sequences)

    # Pad sequences
    padded_src = [seq + [pad_token] * (max_src_len - len(seq)) for seq in src_sequences]
    src_mask = [
        [1] * len(seq) + [0] * (max_src_len - len(seq)) for seq in src_sequences
    ]

    # Convert to tensors
    src = torch.tensor(padded_src, dtype=torch.long, device=device)
    tgt = torch.tensor(tgt_sequences, dtype=torch.long, device=device)
    src_mask = torch.tensor(src_mask, dtype=torch.float, device=device)

    return src, tgt, src_mask


def get_loss(model, criterion, dataset, device, batch_size):
    """
    Calculate loss for a dataset.

    Args:
        model: The model to evaluate
        criterion: Loss function
        dataset: Dataset containing events, questions, and answers
        device: Device to run the model on
        batch_size: Batch size for processing

    Returns:
        Average loss over the dataset
    """
    model.train()
    # src, tgt, src_mask = prepare_batch(dataset["data"], device)
    # output = model(src)

    # seq_lengths = (
    #     src_mask.sum(dim=1).long() - 1
    # )  # -1 because we want the last token position
    # batch_indices = torch.arange(output.size(0), device=device)
    # last_outputs = output[batch_indices, seq_lengths]
    # loss = criterion(last_outputs, tgt.squeeze(1))
    # return loss

    # data = dataset["data"]
    # for i in range(0, len(data), batch_size):
    #     batch_data = data[i : i + batch_size]
    #     src, tgt, src_mask = prepare_batch(batch_data, device)
    #     output = model(src)
    #     seq_lengths = src_mask.sum(dim=1).long() - 1
    #     batch_indices = torch.arange(output.size(0), device=device)
    #     last_outputs = output[batch_indices, seq_lengths]
    #     loss = criterion(last_outputs, tgt.squeeze(1))

    total_loss = 0.0
    data = dataset["data"]

    # Process in batches
    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        src, tgt, src_mask = prepare_batch(batch_data, device)
        output = model(src)

        # Get the last non-padding output for each sequence
        seq_lengths = (
            src_mask.sum(dim=1).long() - 1
        )  # -1 because we want the last token position
        batch_indices = torch.arange(output.size(0), device=device)
        last_outputs = output[batch_indices, seq_lengths]
        batch_loss = criterion(last_outputs, tgt.squeeze(1))
        total_loss += batch_loss * batch_size

    return total_loss / len(data)


@torch.no_grad()
def loss_err(model, criterion, dataset, device, batch_size=100):
    """
    Calculate loss and error rate for a dataset.

    Args:
        model: The model to evaluate
        criterion: Loss function
        dataset: Dataset containing events, questions, and answers
        device: Device to run the model on
        batch_size: Batch size for processing

    Returns:
        Tuple of (loss, error rate)
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    data = dataset["data"]

    # Process in batches
    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        src, tgt, src_mask = prepare_batch(batch_data, device)

        # Forward pass
        output = model(src)

        # Get the last non-padding output for each sequence
        seq_lengths = (
            src_mask.sum(dim=1).long() - 1
        )  # -1 because we want the last token position
        batch_indices = torch.arange(output.size(0), device=device)
        last_outputs = output[batch_indices, seq_lengths]

        # Calculate loss
        loss = criterion(last_outputs, tgt.squeeze(1))

        # Calculate accuracy - compare predictions with targets
        predictions = last_outputs.argmax(dim=-1)
        targets = tgt.squeeze(1)

        # Count correct predictions
        correct = (predictions == targets).sum().item()

        total_loss += loss.item() * len(batch_data)
        total_correct += correct
        total_samples += len(batch_data)

    avg_loss = torch.tensor(total_loss / len(data), device=device)
    error_rate = torch.tensor(1.0 - (total_correct / total_samples), device=device)

    return avg_loss, error_rate


def train_model(
    model,
    config,
    optimizer,
    scheduler,
):
    num_epoch = config.num_epoch

    generator = ScenarioGenerator(
        num_scenarios=config.test_num_scenarios,  # 100,
        min_agents=config.test_min_agents,  # 1,
        max_agents=config.test_max_agents,  # 5,
        num_objects=config.test_num_objects,  # 1,
        min_chain_length=config.test_min_chain_length,  # 1,
        max_chain_length=config.test_max_chain_length,  # 3,
        move_probability=config.test_move_probability,  # 0.8,
        num_questions_per_scenario=config.test_num_questions_per_scenario,  # 10,
        seed=config.test_seed,  # 42,
    )
    test_dataset = generator.generate_dataset()
    print("TEST", len(test_dataset["data"]))
    generator.save_dataset(test_dataset, f"{config.out_dir}/test.json")

    generator = ScenarioGenerator(
        num_scenarios=config.test_ood_num_scenarios,  # 100,
        min_agents=config.test_ood_min_agents,  # 1,
        max_agents=config.test_ood_max_agents,  # 5,
        num_objects=config.test_ood_num_objects,  # 1,
        min_chain_length=config.test_ood_min_chain_length,  # 3,
        max_chain_length=config.test_ood_max_chain_length,
        move_probability=config.test_ood_move_probability,  # 0.8,
        num_questions_per_scenario=config.test_ood_num_questions_per_scenario,  # 10,
        seed=config.test_ood_seed,  # 42,
    )
    test_ood_dataset = generator.generate_dataset()
    print("TEST_OOD", len(test_ood_dataset["data"]))
    generator.save_dataset(test_ood_dataset, f"{config.out_dir}/test_ood.json")

    criterion = (
        nn.CrossEntropyLoss(label_smoothing=0.1)
        if config.label_smoothing
        else nn.CrossEntropyLoss()
    )

    err_arr = np.zeros((num_epoch, 6))
    err_arr_json = []

    if config.wandb_log:
        import wandb
        from datetime import datetime

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        wandb.init(
            project="symbolic_tom",
            config=config.__dict__,
            id=run_id,
            name=f"run-{run_id}",
        )

    for epoch in range(num_epoch):
        model.train()

        generator = ScenarioGenerator(
            num_scenarios=config.train_num_scenarios,
            min_agents=config.train_min_agents,
            max_agents=config.train_max_agents,
            num_objects=config.train_num_objects,
            min_chain_length=config.train_min_chain_length,
            max_chain_length=config.train_max_chain_length,
            move_probability=config.train_move_probability,
            num_questions_per_scenario=config.train_num_questions_per_scenario,
            seed=config.train_seed,
        )
        train_dataset = generator.generate_dataset()
        if config.train_seed is not None and not os.path.exists(
            f"{config.out_dir}/train.json"
        ):
            # train seed not null implies we can using a fixed train set
            generator.save_dataset(train_dataset, f"{config.out_dir}/train.json")
            print("TRAIN", len(train_dataset["data"]))

        optimizer.zero_grad()
        loss = get_loss(
            model,
            criterion,
            train_dataset,
            device=config.device,
            batch_size=config.batch_size,
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            loss_train, train_err = loss_err(
                model, criterion, train_dataset, device=config.device
            )
            loss_test, test_err = loss_err(
                model, criterion, test_dataset, device=config.device
            )
            loss_test_ood, test_err_ood = loss_err(
                model, criterion, test_ood_dataset, device=config.device
            )

        err_arr[epoch, :] = [
            loss_train.item(),
            train_err.item(),
            loss_test.item(),
            test_err.item(),
            loss_test_ood.item(),
            test_err_ood.item(),
        ]

        if config.wandb_log:
            wandb.log(
                {
                    "train/loss": loss_train.item(),
                    "train/error": train_err.item(),
                    "test/loss": loss_test.item(),
                    "test/error": test_err.item(),
                    "ood/loss": loss_test_ood.item(),
                    "ood/error": test_err_ood.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        err_arr_json += [
            {
                "epoch": epoch,
                "loss_train": loss_train.item(),
                "err_train": train_err.item(),
                "loss_test": loss_test.item(),
                "err_test": test_err.item(),
                "loss_ood": loss_test_ood.item(),
                "err_ood": test_err_ood.item(),
            }
        ]

        if err_arr[epoch, 5] > 0.05:
            if config.print_output and (epoch + 1) % config.n_step_print_output == 0:
                print(
                    f"----> Epoch: {epoch+1:>5}, Train Loss: {loss.item():.3f}, Train Error: {train_err:.3f}, Test Error: {test_err:.3f}, OOD Error: {test_err_ood:.3f}"
                )

        if (1 + epoch) % (config.num_epoch // config.n_save) == 0 or (
            config.up_to_first_save
            and (1 + epoch)
            in [
                np.power(2, k)
                for k in range(int(np.log2(config.num_epoch // config.n_save)))
            ]
        ):
            out_path = os.path.join(config.out_dir, f"ckpt_{epoch + 1}.pt")
            torch.save(model.state_dict(), out_path)

    return model, err_arr, err_arr_json
