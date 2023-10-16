import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from utils import save_image, save_image_ori
# from torchvision.utils import save_image, make_grid

from evaluator import evaluation_model

torch.backends.cudnn.benchmark = True

label_type = torch.FloatTensor

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    # nose schedule : linear
    # TODO : can change to cosine
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # xt(x0, ε) = √(α ̄t)*x0 + √(1 − α ̄t)*ε 
    # add noise to image in a time
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # xT ∼ N(0,I) 
            # create an initial image sampling from normal distribution
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # for t = T,...,1 do
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                labels = labels.to(self.device)
                # εθ(xt,t) : model's output, the predicted noise
                predicted_noise = model(x, t, labels)
                # if use classifier-free guidance : 
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    # do linear interpolation between unconditional & conditional predicted noise
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                # σt2 = βt
                beta = self.beta[t][:, None, None, None]
                # z ∼ N(0,I) if t > 1, else z = 0
                # last step don't need noise, cuz it's the final outcome
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                # xt−1 = (1/√αt) * [xt − (1-αt)/(√1−α ̄t)*εθ(xt,t)] +σtz 
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        # let x be 0 ~ 1
        x = (x.clamp(-1, 1) + 1) / 2
        # => valid pixel range 
        x = (x * 255).type(torch.uint8)
        return x




def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_loader,test_loader,new_test_loader = get_data(args)

    if args.retrain:
        logging.info("============== Retrain ==============")
        model = UNet_conditional(num_classes=args.num_classes).to(device)
        ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
        model.load_state_dict(ckpt)

        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        ema_ckpt = torch.load("./models/DDPM_conditional/ema_ckpt.pt")
        ema_model.load_state_dict(ema_ckpt)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        optim_ckpt = torch.load("./models/DDPM_conditional/optim.pt")
        optimizer.load_state_dict(optim_ckpt)

    else:
        model = UNet_conditional(num_classes=args.num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)


    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_loader)
    l_test = len(test_loader)
    l_new_test = len(new_test_loader)
    ema = EMA(0.995)
    

    evaluator = evaluation_model()
    max_acc = 1.2
    best_model = None
    best_ema_model = None


    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            # t ∼ Uniform({1,...,T})
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # x0 ∼ q(x0) , ε ∼ N(0,I)
            x_t, noise = diffusion.noise_images(images, t)
            # classifier-free guidance
            if np.random.random() < 0.1:
                labels = None
            # model : Unet, input noisy_image and t, output predicted noise εθ
            predicted_noise = model(x_t, t, labels)
            # (ε−εθ)^2
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"retrain_ckpt.pt"))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"retrain_ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"retrain_optim.pt"))
            

        
def test(args):
    model = UNet_conditional(num_classes=args.num_classes).to(args.device)
    #ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    ckpt = torch.load("./models/DDPM_retrain/retrain_ckpt.pt")
    model.load_state_dict(ckpt)

    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    #ema_ckpt = torch.load("./models/DDPM_conditional/ema_ckpt.pt")
    ema_ckpt = torch.load("./models/DDPM_retrain/retrain_ema_ckpt.pt")
    ema_model.load_state_dict(ema_ckpt)

    train_loader,test_loader,new_test_loader = get_data(args)
    diffusion = Diffusion(img_size=args.image_size, device=args.device)
    #logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_loader)
    l_test = len(test_loader)
    l_new_test = len(new_test_loader)
    evaluator = evaluation_model()

    model.eval()
    ema_model.eval()
    test_sampled_images = []
    test_ema_sampled_images = []
    test_label = []
    new_test_sampled_images = []
    new_test_ema_sampled_images = []
    new_test_label = []
    with torch.no_grad():
        # 2 images a time => total 32 images so total process 16 times
        for i, label in enumerate(test_loader):
            #sampled_images = diffusion.sample(model, n=args.batch_size, labels=label)
            ema_sampled_images = diffusion.sample(ema_model, n=args.batch_size, labels=label)
            #test_sampled_images.append(sampled_images)
            test_ema_sampled_images.append(ema_sampled_images)
            test_label.append(label)

        for i in range(16):
            #t_s_i = torch.cat((test_sampled_images))
            t_e_s_i = torch.cat((test_ema_sampled_images))
            t_l = torch.cat((test_label))


        #test_score = evaluator.eval(t_s_i, t_l)
        ema_test_score = evaluator.eval(t_e_s_i, t_l)

        for i, label in enumerate(new_test_loader):
            #sampled_images = diffusion.sample(model, n=args.batch_size, labels=label)
            ema_sampled_images = diffusion.sample(ema_model, n=args.batch_size, labels=label)
            #new_test_sampled_images.append(sampled_images)
            new_test_ema_sampled_images.append(ema_sampled_images)
            new_test_label.append(label)

        for i in range(16):
            #n_t_s_i = torch.cat((new_test_sampled_images))
            n_t_e_s_i = torch.cat((new_test_ema_sampled_images))
            n_t_l = torch.cat((new_test_label))

        # evaluate total 32 images
        #new_test_score = evaluator.eval(n_t_s_i, n_t_l)
        ema_new_test_score = evaluator.eval(n_t_e_s_i, n_t_l)

    #test_acc = float("{:.1f}".format(test_score)) + float("{:.1f}".format(new_test_score))
    ema_test_acc = float("{:.1f}".format(ema_test_score)) + float("{:.1f}".format(ema_new_test_score))


    # print(" test score:{:.2f}, new test score:{:.2f}, ema test:{:.2f}, ema new test:{:.2f}"\
    #     .format(test_score, new_test_score,ema_test_score,ema_new_test_score))
    print(" ema test:{:.2f}, ema new test:{:.2f}"\
        .format(ema_test_score,ema_new_test_score))

    #save_image_ori(t_s_i, os.path.join("test_result", f"t_s_i.jpg"))
    save_image_ori(t_e_s_i, os.path.join("test_result",  f"t_e_s_i.jpg"))
    #save_image_ori(n_t_s_i, os.path.join("test_result",  f"n_t_s_i.jpg"))
    save_image_ori(n_t_e_s_i, os.path.join("test_result", f"n_t_e_s_i.jpg"))




def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--weight_dir', default='./weight', type=str, help='weight direction')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()
    args.run_name = "DDPM_retrain"
    args.epochs = 55
    args.batch_size = 2
    args.image_size = 64
    args.num_classes = 24
    #args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
    args.device = "cuda"
    args.lr = 3e-4
    torch.cuda.empty_cache()
    os.makedirs(args.weight_dir, exist_ok=True)

    if args.test:
        #args.run_name = "DDPM_test"
        test(args)
    else:
        train(args)

if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

