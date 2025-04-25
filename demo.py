import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
from MCFNet import MCFNet
import numpy as np
import time
import os
from utils import train_patch, setup_seed, output_metric, print_args, train_epoch, valid_epoch
import logging
from thop import profile

All_OA = []
All_AA = []
All_Kappa = []
All_CA = []



def train_1times(number):
# # setup_seed(args.seed)
#     if args.dataset == 'Houston':
#         num_classes = 15
#         DataPath1 = './mofeng/work3/code/MCFNet/data/Houston/Houston.mat'
#         DataPath2 = './mofeng/work3/code/MCFNet/data/Houston/LiDAR.mat'
#         LabelPath_10TIMES = './mofeng/mofeng/Desktop/work3/code/MCFNet/data/Houston/20/train_test_gt_{}.mat'.format(number)
#         Data1 = loadmat(DataPath1)['img']
#         Data2 = loadmat(DataPath2)['img']
#     elif args.dataset == 'Augsburg':
#         num_classes = 7
#         DataPath1 = './mofeng/work3/code/MCFNet/data/Augsburg/data_HS_LR.mat'
#         DataPath2 = './mofeng/work3/code/MCFNet/data/Augsburg/data_DSM.mat'
#         Data1 = loadmat(DataPath1)['data_HS_LR']
#         Data2 = loadmat(DataPath2)['data_DSM']
#         LabelPath_10TIMES = './mofeng/work3/code/MCFNet/data/Augsburg/20/train_test_gt_{}.mat'.format(number)
#     elif args.dataset == 'Muufl':
#         num_classes = 11
#         DataPath1 = './mofeng/work3/code/MCFNet/data/Muufl/hsi.mat'
#         DataPath2 = './mofeng/work3/code/MCFNet/data/Muufl/lidar_DEM.mat'
#         Data1 = loadmat(DataPath1)['hsi']
#         Data2 = loadmat(DataPath2)['lidar']
#         LabelPath_10TIMES = './mofeng/work3/code/MCFNet/data/Muufl/20/train_test_gt_{}.mat'.format(number)
#     elif args.dataset == 'Trento':
#         num_classes = 9
#         DataPath1 = './mofeng/work3/code/MCFNet/data/Trento/HSI.mat'
#         DataPath2 = './mofeng/work3/code/MCFNet/data/Trento/LiDAR.mat'
#         Data1 = loadmat(DataPath1)['HSI']
#         Data2 = loadmat(DataPath2)['LiDAR']
#         LabelPath_10TIMES = './mofeng/work3/code/MCFNet/data/Trento/20/train_test_gt_{}.mat'.format(number)
#     else:
#         raise "Requires correct dataset name!"
# setup_seed(args.seed)
    if args.dataset == 'Houston':
        num_classes = 15
        DataPath1 = 'C:/Users/mofen/Desktop/work3/code/24_12_06MWNet/data/Houston/Houston.mat'
        DataPath2 = './mofeng/work3/code/MCFNet/data/Houston/LiDAR.mat'
        LabelPath_10TIMES = './mofeng/mofeng/Desktop/work3/code/MCFNet/data/Houston/20/train_test_gt_{}.mat'.format(number)
        Data1 = loadmat(DataPath1)['img']
        Data2 = loadmat(DataPath2)['img']
    elif args.dataset == 'Augsburg':
        num_classes = 7
        DataPath1 = './mofeng/work3/code/MCFNet/data/Augsburg/data_HS_LR.mat'
        DataPath2 = './mofeng/work3/code/MCFNet/data/Augsburg/data_DSM.mat'
        Data1 = loadmat(DataPath1)['data_HS_LR']
        Data2 = loadmat(DataPath2)['data_DSM']
        LabelPath_10TIMES = './mofeng/work3/code/MCFNet/data/Augsburg/20/train_test_gt_{}.mat'.format(number)
    elif args.dataset == 'Muufl':
        num_classes = 11
        DataPath1 = './mofeng/work3/code/MCFNet/data/Muufl/hsi.mat'
        DataPath2 = './mofeng/work3/code/MCFNet/data/Muufl/lidar_DEM.mat'
        Data1 = loadmat(DataPath1)['hsi']
        Data2 = loadmat(DataPath2)['lidar']
        LabelPath_10TIMES = './mofeng/work3/code/MCFNet/data/Muufl/20/train_test_gt_{}.mat'.format(number)
    elif args.dataset == 'Trento':
        num_classes = 9
        DataPath1 = 'C:/Users/mofen/Desktop/work3/code/24_12_06MWNet/data/Trento/HSI.mat'
        DataPath2 = 'C:/Users/mofen/Desktop/work3/code/24_12_06MWNet/data/Trento/LiDAR.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
        LabelPath_10TIMES = 'C:/Users/mofen/Desktop/work3/code/24_12_06MWNet/data/Trento/20/train_test_gt_{}.mat'.format(number)
    else:
        raise "Requires correct dataset name!"
    
    TrLabel_10TIMES = loadmat(LabelPath_10TIMES)['train_data']  # 349*1905
    TsLabel_10TIMES = loadmat(LabelPath_10TIMES)['test_data']  # 349*1905
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    patchsize1 = args.patches1  # input spatial size for 2D-CNN
    pad_width1 = np.floor(patchsize1 / 2)
    pad_width1 = int(pad_width1)  # 8
    patchsize2 = args.patches2  # input spatial size for 2D-CNN
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = int(pad_width2)  # 8
    patchsize3 = args.patches3  # input spatial size for 2D-CNN
    pad_width3 = np.floor(patchsize3 / 2)
    pad_width3 = int(pad_width3)  # 8
    patchsize4 = args.patches4  # input spatial size for 2D-CNN
    pad_width4 = np.floor(patchsize4 / 2)
    pad_width4 = int(pad_width4)  # 8
    TrainPatch11, TrainPatch21, TrainLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TrLabel_10TIMES)
    TestPatch11, TestPatch21, TestLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
    TrainPatch12, TrainPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TrLabel_10TIMES)
    TestPatch12, TestPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
    TrainPatch13, TrainPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TrLabel_10TIMES)
    TestPatch13, TestPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)
    TrainPatch14, TrainPatch24, _ = train_patch(Data1, Data2, patchsize4, pad_width4, TrLabel_10TIMES)
    TestPatch14, TestPatch24, _ = train_patch(Data1, Data2, patchsize4, pad_width4, TsLabel_10TIMES)
    train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainPatch12, TrainPatch22, TrainPatch13, TrainPatch23, TrainPatch14, TrainPatch24, TrainLabel)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23, TestPatch14, TestPatch24, TestLabel)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    logger.info("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    logger.info("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    # -------------------------------------------------------------------------------
    # create model
    model = MCFNet(l1=band1, l2=band2, patch_size=args.patches1, num_classes=args.num_classes,
                fae_embed_dim=args.fae_embed_dim, attn_kernel_size=args.attn_kernel_size, wavename=args.wavename)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------
    # train and test
    if args.flag_test == 'train':
        BestAcc = 0
        val_acc = []
        logger.info("start training")
        tic = time.time()
        for epoch in range(args.epoches):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
            OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
            logger.info("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                  .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
            scheduler.step()

            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, test_loader, criterion)
                OA2, AA2, Kappa2, CA2 = output_metric(tar_v, pre_v)
                val_acc.append(OA2)
                logger.info("Every 20 epochs' records:")
                logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
                logger.info(CA2)
                if OA2 > BestAcc:
                    torch.save(model.state_dict(), './MCFNet.pkl')
                    BestAcc = OA2

        toc = time.time()
        model.eval()
        model.load_state_dict(torch.load('./MCFNet.pkl'))
        tar_v, pre_v = valid_epoch(model, test_loader, criterion)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)

        All_OA.append(OA)
        All_AA.append(AA)
        All_Kappa.append(Kappa)
        All_CA.append(CA)

        logger.info("Final records:")
        logger.info("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
        logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        logger.info(CA)
        logger.info("Running Time: {:.2f}".format(toc - tic))
        logger.info("**************************************************")
        logger.info("Parameter:")
        logger.info(vars(args))


if __name__ == '__main__':
    
    for i in range(0,1):
        # Trento
        if i == 0:
            # -------------------------------------------------------------------------------
            # Parameter Setting
            parser = argparse.ArgumentParser("MCFNet")
            parser.add_argument('--gpu_id', default='0', help='gpu id')
            parser.add_argument('--seed', type=int, default=0, help='number of seed')
            parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
            parser.add_argument('--epoches', type=int, default=500, help='epoch number')  # Muufl 200
            parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # diffGrad 1e-3
            parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
            parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
            parser.add_argument('--num_classes', choices=[11, 6, 15, 7], default=6, help='number of classes')
            parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
            parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
            parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')  #base size
            parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
            parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
            parser.add_argument('--patches4', type=int, default=32, help='number3 of patches')
            parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston', 'Augsburg'], default='Trento', help='dataset to use')
            parser.add_argument('--wavename', type=str, default='bior1.1',help='type of wavelet')
            parser.add_argument('--attn_kernel_size', type=int, default=9,help='')
            parser.add_argument('--fae_embed_dim', type=int, default=128,help='number of channels in fae inpu data')
            args = parser.parse_args()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            '''log'''
            logger = logging.getLogger("Trainlog")
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler("cls_logs/{}/{}.log".format(args.dataset, args.dataset))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            setup_seed(args.seed)
            tic = time.time()
            for i in range(1,11):
                logger.info('-----------------this is the {}----------------------'.format(i))
                train_1times(i)
            toc = time.time()
            if All_OA:
                avg_OA = round(np.mean(All_OA) * 100, 2)
                avg_AA = round(np.mean(All_AA) * 100, 2)
                avg_Kappa = round(np.mean(All_Kappa), 4)
                std_OA = round(np.std(All_OA) * 100, 2)
                std_AA = round(np.std(All_AA) * 100, 2)
                std_Kappa = round(np.std(All_Kappa), 4)

                # 计算所有CA的平均值
                avg_CA = np.mean(All_CA, axis=0)

                logger.info('-----------average-----------')
                logger.info("OA: {:.4f}'+-'{:.4f} | AA: {:.4f}'+-'{:.4f} | Kappa: {:.4f}'+-'{:.4f}".format(avg_OA, std_OA, avg_AA, std_AA, avg_Kappa, std_Kappa))
                logger.info(avg_CA)
                logger.info("Running Time: {:.2f}s".format((toc - tic)/10))
            else:
                print("No valid results to calculate averages.")
        # Muufl
        if i == 1:
            # -------------------------------------------------------------------------------
            # Parameter Setting
            parser = argparse.ArgumentParser("MCFNet")
            parser.add_argument('--gpu_id', default='0', help='gpu id')
            parser.add_argument('--seed', type=int, default=0, help='number of seed')
            parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
            parser.add_argument('--epoches', type=int, default=500, help='epoch number')  # Muufl 200
            parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')  # diffGrad 1e-3
            parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
            parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
            parser.add_argument('--num_classes', choices=[11, 6, 15, 7], default=11, help='number of classes')
            parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
            parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
            parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')  #base size
            parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
            parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
            parser.add_argument('--patches4', type=int, default=32, help='number3 of patches')
            parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston', 'Augsburg'], default='Muufl', help='dataset to use')
            parser.add_argument('--wavename', type=str, default='db2',help='type of wavelet')
            parser.add_argument('--attn_kernel_size', type=int, default=9,help='')
            parser.add_argument('--fae_embed_dim', type=int, default=64,help='number of channels in fae inpu data')
            args = parser.parse_args()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            '''log'''
            logger = logging.getLogger("Trainlog")
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler("cls_logs/{}/{}.log".format(args.dataset, args.dataset))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            setup_seed(args.seed)
            tic = time.time()
            for i in range(1,11):
                logger.info('-----------------this is the {}----------------------'.format(i))
                train_1times(i)
            toc = time.time()
            if All_OA:
                avg_OA = round(np.mean(All_OA) * 100, 2)
                avg_AA = round(np.mean(All_AA) * 100, 2)
                avg_Kappa = round(np.mean(All_Kappa), 4)
                std_OA = round(np.std(All_OA) * 100, 2)
                std_AA = round(np.std(All_AA) * 100, 2)
                std_Kappa = round(np.std(All_Kappa), 4)

                # 计算所有CA的平均值
                avg_CA = np.mean(All_CA, axis=0)

                logger.info('-----------average-----------')
                logger.info("OA: {:.4f}'+-'{:.4f} | AA: {:.4f}'+-'{:.4f} | Kappa: {:.4f}'+-'{:.4f}".format(avg_OA, std_OA, avg_AA, std_AA, avg_Kappa, std_Kappa))
                logger.info(avg_CA)
                logger.info("Running Time: {:.2f}s".format((toc - tic)/10))
            else:
                print("No valid results to calculate averages.")
        # Augsburg
        if i == 2:
            # -------------------------------------------------------------------------------
            # Parameter Setting
            parser = argparse.ArgumentParser("MCFNet")
            parser.add_argument('--gpu_id', default='0', help='gpu id')
            parser.add_argument('--seed', type=int, default=0, help='number of seed')
            parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
            parser.add_argument('--epoches', type=int, default=500, help='epoch number')  # Muufl 200
            parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')  # diffGrad 1e-3
            parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
            parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
            parser.add_argument('--num_classes', choices=[11, 6, 15, 7], default=7, help='number of classes')
            parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
            parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
            parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')  #base size
            parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
            parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
            parser.add_argument('--patches4', type=int, default=32, help='number3 of patches')
            parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston', 'Augsburg'], default='Augsburg', help='dataset to use')
            parser.add_argument('--wavename', type=str, default='haar',help='type of wavelet')
            parser.add_argument('--attn_kernel_size', type=int, default=9,help='')
            parser.add_argument('--fae_embed_dim', type=int, default=64,help='number of channels in fae inpu data')
            args = parser.parse_args()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            '''log'''
            logger = logging.getLogger("Trainlog")
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler("cls_logs/{}/{}.log".format(args.dataset, args.dataset))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            setup_seed(args.seed)
            tic = time.time()
            for i in range(1,11):
                logger.info('-----------------this is the {}----------------------'.format(i))
                train_1times(i)
            toc = time.time()
            if All_OA:
                avg_OA = round(np.mean(All_OA) * 100, 2)
                avg_AA = round(np.mean(All_AA) * 100, 2)
                avg_Kappa = round(np.mean(All_Kappa), 4)
                std_OA = round(np.std(All_OA) * 100, 2)
                std_AA = round(np.std(All_AA) * 100, 2)
                std_Kappa = round(np.std(All_Kappa), 4)

                # 计算所有CA的平均值
                avg_CA = np.mean(All_CA, axis=0)

                logger.info('-----------average-----------')
                logger.info("OA: {:.4f}'+-'{:.4f} | AA: {:.4f}'+-'{:.4f} | Kappa: {:.4f}'+-'{:.4f}".format(avg_OA, std_OA, avg_AA, std_AA, avg_Kappa, std_Kappa))
                logger.info(avg_CA)
                logger.info("Running Time: {:.2f}s".format((toc - tic)/10))
            else:
                print("No valid results to calculate averages.")
        
        # Houston
        if i == 3:
            # -------------------------------------------------------------------------------
            # Parameter Setting
            parser = argparse.ArgumentParser("MCFNet")
            parser.add_argument('--gpu_id', default='0', help='gpu id')
            parser.add_argument('--seed', type=int, default=0, help='number of seed')
            parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
            parser.add_argument('--epoches', type=int, default=500, help='epoch number')  # Muufl 200
            parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')  # diffGrad 1e-3
            parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
            parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
            parser.add_argument('--num_classes', choices=[11, 6, 15], default=15, help='number of classes')
            parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
            parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
            parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')  #base size
            parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
            parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
            parser.add_argument('--patches4', type=int, default=32, help='number3 of patches')
            parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston', 'Augsburg'], default='Houston', help='dataset to use')
            parser.add_argument('--wavename', type=str, default='haar',help='type of wavelet')
            parser.add_argument('--attn_kernel_size', type=int, default=9,help='')
            parser.add_argument('--fae_embed_dim', type=int, default=64,help='number of channels in fae inpu data')
            args = parser.parse_args()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            '''log'''
            logger = logging.getLogger("Trainlog")
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler("cls_logs/{}/{}.log".format(args.dataset, args.dataset))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            setup_seed(args.seed)
            tic = time.time()
            for i in range(1,11):
                logger.info('-----------------this is the {}----------------------'.format(i))
                train_1times(i)
            toc = time.time()
            if All_OA:
                avg_OA = round(np.mean(All_OA) * 100, 2)
                avg_AA = round(np.mean(All_AA) * 100, 2)
                avg_Kappa = round(np.mean(All_Kappa), 4)
                std_OA = round(np.std(All_OA) * 100, 2)
                std_AA = round(np.std(All_AA) * 100, 2)
                std_Kappa = round(np.std(All_Kappa), 4)

                # 计算所有CA的平均值
                avg_CA = np.mean(All_CA, axis=0)

                logger.info('-----------average-----------')
                logger.info("OA: {:.4f}'+-'{:.4f} | AA: {:.4f}'+-'{:.4f} | Kappa: {:.4f}'+-'{:.4f}".format(avg_OA, std_OA, avg_AA, std_AA, avg_Kappa, std_Kappa))
                logger.info(avg_CA)
                logger.info("Running Time: {:.2f}s".format((toc - tic)/10))
            else:
                print("No valid results to calculate averages.")



