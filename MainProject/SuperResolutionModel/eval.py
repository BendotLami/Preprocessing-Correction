from .utils import *
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .datasets import SRDataset
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = "./"
test_data_names = ["Set5"]

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Load model, either the SRResNet or the SRGAN
# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# srresnet.eval()
# model = srresnet
model = None

image_counter = 0


def srgan_load_model(checkpoint):
    global model
    srgan_generator = torch.load(checkpoint)['generator'].to(device)
    srgan_generator.eval()
    model = srgan_generator

def save_images(images, lr_images):
    images = images.cpu()
    lr_images = lr_images.cpu()
    # images =
    global image_counter
    for i in range(images.shape[0]):
        real = images[i].numpy().transpose(1, 2, 0)
        # print(real.dtype, np.max(real), np.min(real))
        real = (real + 1.) / 2.
        real = np.clip(real, 0, 1)

        lr_real = lr_images[i].numpy().transpose(1, 2, 0)
        # print(lr_real.dtype, np.max(lr_real), np.min(lr_real))
        lr_real = (lr_real + 2.) / 4.
        lr_real = np.clip(lr_real, 0, 1)

        plt.imsave(str("./srgan_outputs/" + "output_" + str(i) + "_batch_" + str(image_counter) + ".jpg"), real)
        plt.imsave(str("./srgan_outputs/" + "output_" + str(i) + "_batch_" + str(image_counter) + "_lr.jpg"), lr_real)
    image_counter += 1
    print("Done printing! count:", image_counter)


def SRgan_forward_pass(img):
    global model
    model.eval().to(device)
    if len(img.size()) < 4:  # if we want to forward pass a single image
        img = img[None, :, :, :]
    img = img.to(device)
    with torch.no_grad():
        rtn_img = model(img)
    return rtn_img


# print("\n")

if __name__ == "__main__":
    srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    srresnet.eval()
    model = srresnet

    # Evaluate
    for test_data_name in test_data_names:
        print("\nFor %s:\n" % test_data_name)

        # Custom dataloader
        test_dataset = SRDataset(data_folder,
                                 split='test',
                                 crop_size=0,
                                 scaling_factor=4,
                                 lr_img_type='imagenet-norm',
                                 hr_img_type='[-1, 1]',
                                 test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                                  pin_memory=True)

        # Keep track of the PSNRs and the SSIMs across batches
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        # Prohibit gradient computation explicitly because I had some problems with memory
        with torch.no_grad():
            # Batches
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                # Move to default device
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

                # Calculate PSNR and SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)  # (w, h), in y-channel
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
                # psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                #                                data_range=255.)
                # ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                #                              data_range=255.)
                # PSNRs.update(psnr, lr_imgs.size(0))
                # SSIMs.update(ssim, lr_imgs.size(0))

                save_images(sr_imgs, lr_imgs)

        # Print average PSNR and SSIM
        # print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        # print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

