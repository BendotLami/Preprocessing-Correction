import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2

from preprocess_model import *

BATCH_SIZE = 64

def rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - torch.abs(torch.abs(x - y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(torch.argmax(y_true), torch.argmax(y_pred))
    return torch.mean(torch.abs(diff).float())


class ModelAgentColorCorrection(object):
    def __init__(self, dataset):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.model = ColorCorrectionNet().to(self.device)
        self.generator = RotationGenerator().to(self.device)
        self.discriminator = RotationDiscriminator().to(self.device)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=800000, gamma=0.1)

        self.dataset = dataset

        transforms = torch.nn.Sequential(
            torchvision.transforms.RandomRotation(degrees=20, fill=0, resample=0)
        )
        self.scripted_transforms = torch.jit.script(transforms)

        self.generator_classification_lambda = 0.01
        self.generator_geometric_lambda = 1

        self.start_rotation_sigma = 0.1

    def train(self):
        train_size = int(0.9 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.BCELoss()

        for epoch in range(100):
            for batch_data in train_data_loader:
                batch_data = batch_data.to(self.device)
                batch_data_classification = torch.ones([batch_data.shape[0], 1]).to(self.device)
                self.generator_optimizer.zero_grad()

                # generator training
                fake_input_batch = self.scripted_transforms(batch_data).to(self.device)
                fake_input_classification = torch.ones([batch_data.shape[0], 1]).to(self.device)
                generator_start_rotation = torch.normal(torch.zeros((batch_data.shape[0], 6)), torch.ones((batch_data.shape[0], 6)) * self.start_rotation_sigma).to(self.device)
                fake_output_batch, matrix = self.generator(fake_input_batch, generator_start_rotation)
                fake_output_batch = fake_output_batch.to(self.device)
                matrix = matrix.to(self.device)

                fake_output_rating = self.discriminator(fake_output_batch).to(self.device)
                classification_loss_generator = criterion(fake_output_rating, fake_input_classification)
                geometric_loss = torch.sum(matrix**2)

                print("gene-class: ", classification_loss_generator.data)
                print("gene-geo: ", geometric_loss.data)

                generator_loss = self.generator_classification_lambda * classification_loss_generator \
                                 + self.generator_geometric_lambda * geometric_loss
                generator_loss.backward()
                self.generator_optimizer.step()

                # discriminator
                self.discriminator_optimizer.zero_grad()
                true_discriminator_out = self.discriminator(batch_data).to(self.device)
                true_discriminator_loss = criterion(true_discriminator_out, batch_data_classification)

                generator_discriminator_out = self.discriminator(fake_output_batch.detach()).to(self.device)
                generator_discriminator_loss = criterion(generator_discriminator_out, torch.zeros([batch_data.shape[0], 1]).to(self.device))

                print("disc-true: ", true_discriminator_loss.data)
                print("disc-fake: ", generator_discriminator_loss.data)

                discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                print("Done batch!")

            with torch.no_grad():
                save_idx = 0
                for batch_data_test in test_data_loader:
                    batch_data_test = batch_data_test.to(self.device)
                    data_augmented = self.scripted_transforms(batch_data_test).to(self.device)
                    generator_start_rotation = torch.normal(torch.zeros((batch_data_test.shape[0], 6)), torch.ones((batch_data_test.shape[0], 6)) * self.start_rotation_sigma).to(self.device)

                    img_reconstructed, _ = self.generator(data_augmented, generator_start_rotation)
                    img_reconstructed = img_reconstructed.to(self.device)

                    # print(criterion(img_reconstructed, batch_data_test).data)

                    test_examples_cpu = batch_data_test.cpu()

                    reconstruction_cpu = img_reconstructed.cpu()

                    data_augmented = data_augmented.cpu()

                    for index in range(int(len(img_reconstructed)/10)):
                        img = test_examples_cpu[index].numpy()

                        img_reconstruct = reconstruction_cpu[index].numpy()

                        img_color_augmented = data_augmented[index].numpy()

                        valid_reconstruct_img = np.clip(img_reconstruct, 0, 1)

                        plt.imsave(str("./test_output_rotation/" + str(epoch) + "_" + str(save_idx) + ".jpg"),
                                   np.concatenate((img.transpose(1, 2, 0), img_color_augmented.transpose(1, 2, 0),
                                                   valid_reconstruct_img.transpose(1, 2, 0)), axis=1))

                        save_idx += 1

            torch.save(self.discriminator.state_dict(), str("./Model_Weights/" + "disc_weights" + "_" + str(epoch)))
            torch.save(self.generator.state_dict(), str("./Model_Weights/" + "gen_weights" + "_" + str(epoch)))

            print("Done epoch", epoch, "!")
