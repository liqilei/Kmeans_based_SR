import argparse, time, os
import random

import torch
import pandas as pd
from tqdm import tqdm

import options.options as option
from utils import util
from models.SRModel import SRModel
from data import create_dataloader
from data import create_dataset


def main():
    # get options
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)

    if opt['train']['resume'] is False:
        util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root' and \
                     not key == 'pretrain_G' and not key == 'pretrain_D'))
        option.save(opt)
        opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    else:
        opt = option.dict_to_nonedict(opt)
        if opt['train']['resume_path'] is None:
            raise ValueError("The 'resume_path' does not declarate")

    if opt['exec_debug']:
        NUM_EPOCH = 50
        opt['datasets']['train']['dataroot_HR'] = '/home/ser606/ZhenLi/data/DIV2K/DIV2K_train_HR_debug'
        opt['datasets']['train']['dataroot_LR'] = '/home/ser606/ZhenLi/data/DIV2K/DIV2K_train_LR_debug'
    else:
        NUM_EPOCH = int(opt['num_epochs'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('Number of train images in [%s]: %d' % (dataset_opt['name'], len(train_set)))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if train_loader is None:
        raise ValueError("The training data does not exist")

    if 'coeff' in train_set[0]:
        opt['networks']['G']['num_coeff'] = len(train_set[0]['coeff'])
        opt['train']['num_coeff'] = len(train_set[0]['coeff'])
    # TODO: design an exp that can obtain the location of the biggest error
    solver = SRModel(opt)
    solver.summary(train_set[0]['LR'].size())
    solver.net_init()

    print('[Start Training]')

    start_time = time.time()

    # resume from the latest epoch
    start_epoch = 1
    if opt['train']['resume']:
        checkpoint_path = os.path.join(solver.checkpoint_dir,'checkpoint.pth')
        print('[Loading checkpoint from %s...]'%checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        solver.model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Because the last state had been saved
        solver.optimizer.load_state_dict(checkpoint['optimizer'])
        solver.best_prec = checkpoint['best_prec']
        solver.results = checkpoint['results']
        print('=> Done.')

    # start train
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        # Initialization
        solver.training_loss = 0.0
        if opt['mode'] == 'sr':
            training_results = {'batch_size': 0, 'training_loss': 0.0}
        else:
            pass    # TODO
        train_bar = tqdm(train_loader)

        # Train model
        for iter, batch in enumerate(train_bar):
            solver.feed_data(batch)
            iter_loss = solver.train_step()
            batch_size = batch['LR'].size(0)
            training_results['batch_size'] += batch_size

            if opt['mode'] == 'sr':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            else:
                pass    # TODO

        train_bar.close()
        time_elapse = time.time() - start_time
        print('\n Train Time: %f seconds' % (time_elapse))

        start_time = time.time()
        # validate
        val_results = {'batch_size': 0, 'val_loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}

        if epoch % solver.val_step == 0 and epoch != 0:

            print('[Validating...]')
            start_time = time.time()
            solver.val_loss = 0.0

            val_bar = tqdm(val_loader)

            for iter, batch in enumerate(val_bar):
                solver.feed_data(batch)
                iter_loss = solver.test()
                batch_size = batch['LR'].size(0)
                val_results['batch_size'] += batch_size
                val_results['val_loss'] += iter_loss * batch_size
                if opt['mode'] == 'srgan':
                    pass    # TODO

            time_elapse = time.time() - start_time
            print('\n Valid Time: %f seconds | Valid Loss: %.4f '%(time_elapse, iter_loss))

            #if epoch%solver.log_step == 0 and epoch != 0:
            # tensorboard visualization
            solver.training_loss = training_results['training_loss'] / training_results['batch_size']
            solver.val_loss = val_results['val_loss'] / val_results['batch_size']

            # TODO: I haven't installed tensorflow, because I should install cuda 9.0 first
            # solver.tf_log(epoch)

            # statistics
            if opt['mode'] == 'sr':
                solver.results['training_loss'].append(float(solver.training_loss.data.cpu().numpy()))
                solver.results['val_loss'].append(float(solver.val_loss.data.cpu().numpy()))
            else:
                pass    # TODO

            is_best = False
            if solver.best_prec < solver.results['val_loss'][-1]:
                solver.best_prec = solver.results['val_loss'][-1]
                is_best = True

            solver.save(epoch, is_best)

        # update lr
        solver.update_learning_rate()

    data_frame = pd.DataFrame(
        data={'training_loss': solver.results['training_loss']
            , 'val_loss': solver.results['val_loss']
              },
        index=range(1, NUM_EPOCH+1)
    )
    data_frame.to_csv(os.path.join(solver.results_dir, 'train_results.csv'),
                      index_label='Epoch')


if __name__ == '__main__':
    main()