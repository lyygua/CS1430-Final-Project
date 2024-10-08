import os
import argparse
import tqdm
import torch

import transferattack
from transferattack.utils import *

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='mifgsm', type=str, help='the attack algorithm')
                        # choices=['fgsm', 'ifgsm', 'mifgsm', 'nifgsm', 'vmifgsm', 'vnifgsm', 'pifgsm', 'emifgsm','sgm',
                        #          'pna_patchout','setr','dsm','dwp','sapr',
                        #          'dim', 'tim', 'sim', 'admix', 'bsr', 'gsit','ide','dem','ssm', 'scm', 'pam',
                        #          'taig','tap','ata','ifgssm','fia','ila','naa','aa','atta','rpa','vaifgsm','linbp','aifgtm','rap','iaa','gimifgsm','fmaa','logit', 'trap',
                        #          'lgv','lifgsm','bayes','masked'])
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=32, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--helper_folder',default='./helper',type=str, help='the path to store the helper models')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)
    return parser.parse_args()


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    if not args.eval:
        if args.attack in transferattack.attack_zoo:
            # TODO: move model loading module into atta, iaa attack itself
            # if args.attack in ['atta']:
            #     from transferattack.input_transformation import atta
            #     atta_model = atta.train_ATTA_Model(args.input_dir, args.batchsize, f2l, 10, 10, eval_model=args.model, device='cuda', path=args.helper_folder)
            #     attacker = transferattack.attack_zoo[args.attack.lower()](model,atta_model=atta_model)
            # elif args.attack in ['iaa']:
            #     from transferattack.gradient import iaa
            #     iaa_model = iaa.tune_IAA_Model(args.input_dir, args.batchsize, f2l, eval_model=args.model, device='cuda', tune=True)
            #     attacker = transferattack.attack_zoo[args.attack.lower()](iaa_model)

            # else:
            attacker = transferattack.attack_zoo[args.attack.lower()](model_name = args.model, targeted = args.targeted)
        else:
            raise Exception("Unspported attack algorithm {}".format(args.attack))

        for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
            perturbations = attacker(images, labels)
            save_images(args.output_dir, images+perturbations.cpu(), filenames)
    else:
        asr = dict()
        res = '|'
        # for model_name, model in load_pretrained_model(cnn_model=['resnet18'],vit_model=['pit_b_224']): # use custom models in torchvision,timm
        # for model_name, model in load_pretrained_model(cnn_model_pkg,vit_model_pkg): # use all models in pkg
        # for model_name, model in load_pretrained_model(cnn_model=['resnet101']): # NRP purified_data are evaluated on resnet101
        # for model_name, model in load_pretrained_model(vit_model=tgr_vit_model_list): # use vit model in tgr
        for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
            model = wrap_model(model.eval().cuda())
            for p in model.parameters():
                p.requires_grad = False
            correct, total = 0, 0
            for images, labels, _ in dataloader:
                if args.targeted:
                    labels = labels[1]
                pred = model(images.cuda())
                correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
            if args.targeted: # correct: pred == target_label
                asr[model_name] = (correct / total) * 100
            else: # correct: pred == original_label
                asr[model_name] = (1 - correct / total) * 100
            print(model_name, asr[model_name])
            res += ' {:.1f} |'.format(asr[model_name])

        print(asr)
        print(res)
        # print('Avg ASR: {:.1f}'.format(sum(asr.values()) / len(asr)))
        with open('results_eval.txt', 'a') as f:
            f.write(args.attack + '_' + args.model + res + '\n')


if __name__ == '__main__':
    main()

