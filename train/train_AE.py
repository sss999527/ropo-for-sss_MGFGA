import torch
import torch.nn as nn
import numpy as np
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter
import math
from train.tsne import visualize_with_tsne


def train_AE(train_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
          num_classes, domain_weight, target_weight_init, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
          confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level):
    task_criterion = nn.CrossEntropyLoss().cuda()
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        # Train model locally on source domains
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                     model_list[1:],
                                                                                     classifier_list[1:],
                                                                                     optimizer_list[1:],
                                                                                     classifier_optimizer_list[1:]):

            # check if the source domain is the malicious domain with poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            for i, (image_s, label_s) in enumerate(train_dloader):
                all_features = []
                all_labels = []
                # all_features_t = []
                # all_labels_t = []

                if i >= batch_per_epoch:
                    break
                image_s = image_s.cuda()
                label_s = label_s.long().cuda()
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # each source domain do optimize
                feature_s = model(image_s)
                output_s = classifier(feature_s)
                task_loss_s = task_criterion(output_s, label_s)
                task_loss_s.backward()
                optimizer.step()
                classifier_optimizer.step()
                # break#2
            # break#3


    # Domain adaptation on target domain
    confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
    target_weight = [0, 0]
    consensus_focus_dict = {}
    for i in range(1, len(train_dloader_list)):
        consensus_focus_dict[i] = 0
    success_attack, total_attack = 0, 0
    num_correct_PL, num_PL = 0.0, 0.0
    for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
        if i >= batch_per_epoch:
            break
        optimizer_list[0].zero_grad()
        classifier_optimizer_list[0].zero_grad()
        image_t = image_t.cuda()
        label_t = label_t.cuda()
        # Knowledge Vote
        with torch.no_grad():
            # knowledge_list = [torch.softmax(classifier_list[i](model_list[i](image_t)), dim=1).unsqueeze(1) for
            #                   i in range(1, source_domain_num + 1)]
            knowledge_list = []
            raw_knowledge_list = []
            for i in range(1, source_domain_num + 1):
                res = torch.softmax(classifier_list[i](model_list[i](image_t)), dim=1)
                raw_knowledge_list.append(res)
                knowledge_list.append(res.unsqueeze(1))
            knowledge_list = torch.cat(knowledge_list, 1)
        # _, consensus_knowledge, consensus_weight = knowledge_vote(knowledge_list, confidence_gate,
        #                                                           num_classes=num_classes)
        konwledge_contribution, consensus_knowledge, consensus_weight, abandon, total = fastly_PL_denoise_with_AE(raw_knowledge_list, knowledge_list, num_classes=num_classes, image_t=image_t,
                                                                                   model_list=model_list, classifier_list=classifier_list, epoch=epoch, total_epochs = total_epochs, confidence_gate = confidence_gate)
        num_PL = num_PL + torch.sum(consensus_weight).item()
        _, pred = consensus_knowledge.max(1)
        correct_pred = (label_t == pred).float().cuda()
        num_correct_PL = num_correct_PL + torch.sum(correct_pred * consensus_weight).item()
        success_attack, total_attack = success_attack + abandon , total_attack + total
        target_weight[0] += torch.sum(consensus_weight).item()
        target_weight[1] += consensus_weight.size(0)
        # Perform data augmentation with mixup
        lam = np.random.beta(2, 2)
        batch_size = image_t.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_image = lam * image_t + (1 - lam) * image_t[index, :]
        mixed_consensus = lam * consensus_knowledge + (1 - lam) * consensus_knowledge[index, :]
        feature_t = model_list[0](mixed_image)

        # tsne
        all_features.append(feature_t.detach().cpu().numpy())
        all_labels.append(label_t.detach().cpu().numpy())

        # visualize_with_tsne(feature_s.detach().cpu().numpy(), label_s.detach().cpu().numpy(), list(range(31)))



        output_t = classifier_list[0](feature_t)
        output_t = torch.log_softmax(output_t, dim=1)
        task_loss_t = torch.mean(consensus_weight * torch.sum(-1 * mixed_consensus * output_t, dim=1))
        # Warm Up
        # if epoch < 10:
        if False:
            for i in range(source_domain_num):
                if i==0:
                    knowledge_ensemble = raw_knowledge_list[i] * (1 / source_domain_num)
                else:
                    knowledge_ensemble = knowledge_ensemble + raw_knowledge_list[i] * (1 / source_domain_num)
            feature_t_WarmUp = model_list[0](image_t)
            output_t_WarmUp = classifier_list[0](feature_t_WarmUp)
            output_t_WarmUp = torch.log_softmax(output_t_WarmUp, dim=1)
            task_loss_t_WarmUp = torch.mean(torch.sum(-1 * knowledge_ensemble * output_t_WarmUp, dim=1))
            task_loss_t = task_loss_t + task_loss_t_WarmUp
        task_loss_t.backward()
        optimizer_list[0].step()
        classifier_optimizer_list[0].step()
        # Calculate consensus focus
        # consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
        #                                                  source_domain_num, num_classes)
        for i in range(1, len(train_dloader_list)):
            consensus_focus_dict[i] += konwledge_contribution[i-1]
        # break#4
    print('Epoch :', epoch, '---> success_attack / total_attack :', success_attack, '/', total_attack)
    print('num_correct_PL / num_PL :', num_correct_PL, '/', num_PL, ' = ', num_correct_PL / num_PL)

    # 在每个epoch结束后进行t-SNE可视化
    if epoch == 0:
        all_features_np = np.concatenate(all_features, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        visualize_with_tsne(all_features_np, all_labels_np, list(range(10)), 'tsne_output0.pdf')
    # if epoch == 0:
    #     all_features_t_np = np.concatenate(all_features_t, axis=0)
    #     all_labels_t_np = np.concatenate(all_labels_t, axis=0)
    #     visualize_with_tsne(all_features_t_np, all_labels_t_np, list(range(10)), 'tsne_output0.pdf')
    if epoch == total_epochs - 1:
        all_features_np = np.concatenate(all_features, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        visualize_with_tsne(all_features_np, all_labels_np, list(range(10)), 'tsne_output80.pdf')

    # compute target domain weight
    if epoch == 0:
        target_weight_init = 1.0 * (1.0 - (target_weight[0] / target_weight[1]))
        # target_weight_init = 0.43623333333333336
    target_weight = 1.0 / (1 + source_domain_num) + (target_weight_init - 1.0 / (1 + source_domain_num)) * (epoch / total_epochs)
    epoch_domain_weight = []
    source_total_weight = 1 - target_weight
    need_softmax = True
    if need_softmax:
        total_knowledge_contribution = 0
        for i in range(1, source_domain_num + 1):
            total_knowledge_contribution += consensus_focus_dict[i]
        for i in range(1, source_domain_num + 1):
            consensus_focus_dict[i] = math.exp(consensus_focus_dict[i] / total_knowledge_contribution)
            epoch_domain_weight.append(consensus_focus_dict[i])
    else:
        for i in range(1, source_domain_num + 1):
            epoch_domain_weight.append(consensus_focus_dict[i])
    epoch_domain_weight = [round(source_total_weight * v / sum(epoch_domain_weight), 4) for v in
                           epoch_domain_weight]
    epoch_domain_weight.insert(0, target_weight)
    if epoch == 0:
        domain_weight = epoch_domain_weight
    else:
        domain_weight = update_domain_weight(domain_weight, epoch_domain_weight)
    # Model aggregation and Batchnorm MMD
    federated_average(model_list, domain_weight, batchnorm_mmd=batchnorm_mmd)
    # Recording domain weight in logs
    writer.add_scalar(tag="Train/target_domain_weight", scalar_value=target_weight, global_step=epoch + 1)
    for i in range(0, len(train_dloader_list) - 1):
        writer.add_scalar(tag="Train/source_domain_{}_weight".format(source_domains[i]),
                          scalar_value=domain_weight[i + 1], global_step=epoch + 1)
    print("Target Domain Weight :{}".format(domain_weight[0]))
    print("Target Domain Weight Init :{}".format(target_weight_init))
    print("Source Domains:{}, Domain Weight :{}".format(source_domains, domain_weight[1:]))
    return domain_weight, target_weight_init


def test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes=126,
         top_5_accuracy=True):
    source_domain_losses = [AverageMeter() for i in source_domains]
    target_domain_losses = AverageMeter()
    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.eval()
    for classifier in classifier_list:
        classifier.eval()
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            output_t = classifier_list[0](model_list[0](image_t))
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        task_loss_t = task_criterion(output_t, label_t)
        target_domain_losses.update(float(task_loss_t.item()), image_t.size(0))
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
    writer.add_scalar(tag="Test/target_domain_{}_loss".format(target_domain), scalar_value=target_domain_losses.avg,
                      global_step=epoch + 1)
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    writer.add_scalar(tag="Test/target_domain_{}_accuracy_top1".format(target_domain).format(target_domain),
                      scalar_value=top_1_accuracy_t,
                      global_step=epoch + 1)
    if top_5_accuracy:
        top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/target_domain_{}_accuracy_top5".format(target_domain).format(target_domain),
                          scalar_value=top_5_accuracy_t,
                          global_step=epoch + 1)
        print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                          top_5_accuracy_t))
    else:
        print("Target Domain {} Accuracy {:.3f}".format(target_domain, top_1_accuracy_t))
    # calculate loss, accuracy for source domains
    for s_i, domain_s in enumerate(source_domains):
        tmp_score = []
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            with torch.no_grad():
                output_s = classifier_list[s_i + 1](model_list[s_i + 1](image_s))
            label_onehot_s = torch.zeros(label_s.size(0), num_classes).cuda().scatter_(1, label_s.view(-1, 1), 1)
            task_loss_s = task_criterion(output_s, label_s)
            source_domain_losses[s_i].update(float(task_loss_s.item()), image_s.size(0))
            tmp_score.append(torch.softmax(output_s, dim=1))
            # turn label into one-hot code
            tmp_label.append(label_onehot_s)
        writer.add_scalar(tag="Test/source_domain_{}_loss".format(domain_s), scalar_value=source_domain_losses[s_i].avg,
                          global_step=epoch + 1)
        tmp_score = torch.cat(tmp_score, dim=0).detach()
        tmp_label = torch.cat(tmp_label, dim=0).detach()
        _, y_true = torch.topk(tmp_label, k=1, dim=1)
        _, y_pred = torch.topk(tmp_score, k=5, dim=1)
        top_1_accuracy_s = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
        writer.add_scalar(tag="Test/source_domain_{}_accuracy_top1".format(domain_s), scalar_value=top_1_accuracy_s,
                          global_step=epoch + 1)
        if top_5_accuracy:
            top_5_accuracy_s = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
            writer.add_scalar(tag="Test/source_domain_{}_accuracy_top5".format(domain_s), scalar_value=top_5_accuracy_s,
                              global_step=epoch + 1)


def adapt_test(target_domain, test_dloader_list, model, classifier, num_classes=345, top_5_accuracy=True):
    model.eval()
    classifier.eval()
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t.cuda()
        label_t = label_t.long().cuda()
        with torch.no_grad():
            output_t = classifier(model(image_t))
        label_onehot_t = torch.zeros(label_t.size(0), num_classes).cuda().scatter_(1, label_t.view(-1, 1), 1)
        tmp_score.append(torch.softmax(output_t, dim=1))
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)
    tmp_score = torch.cat(tmp_score, dim=0).detach()
    tmp_label = torch.cat(tmp_label, dim=0).detach()
    _, y_true = torch.topk(tmp_label, k=1, dim=1)
    _, y_pred = torch.topk(tmp_score, k=5, dim=1)
    top_1_accuracy_t = float(torch.sum(y_true == y_pred[:, :1]).item()) / y_true.size(0)
    if top_5_accuracy:
        top_5_accuracy_t = float(torch.sum(y_true == y_pred).item()) / y_true.size(0)
        print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                          top_5_accuracy_t))
    else:
        print("Target Domain {} Accuracy {:.3f}".format(target_domain, top_1_accuracy_t))
