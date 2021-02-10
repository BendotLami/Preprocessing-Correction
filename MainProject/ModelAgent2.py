import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models import *

BATCH_SIZE = 16

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
    diff = angle_difference(y_true, torch.argmax(y_pred))
    return torch.mean(torch.abs(diff).float())


def get_random_angles_array(batch_size):
    angles = ((torch.rand(batch_size) - 0.5) * 90)
    angles = angles.long() % 360

    return angles


class ModelAgentRotationCorrection(object):
    def __init__(self, dataset):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = RotationCorrectionNet().to(self.device)
        self.model_optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, step_size=800000, gamma=0.1)

        self.dataset = dataset


    def train(self):
        train_size = int(0.9 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.NLLLoss()

        for epoch in range(100):
            ## train
            for batch_data in train_data_loader:
                batch_data = batch_data.to(self.device)
                angles = get_random_angles_array(batch_data.shape[0]).to(self.device)
                for i in range(batch_data.shape[0]):
                    batch_data[i] = torchvision.transforms.functional.rotate(batch_data[i], float(angles[i])).to(self.device)

                feat = self.model(batch_data).to(self.device)

                # train_loss = torch.autograd.Variable(angle_error(angles, feat), requires_grad=True)
                # print(feat.dtype, angles_probability.dtype)
                train_loss = criterion(feat, angles)

                self.model_optimizer.zero_grad()
                print("angle error: ", angle_error(angles, feat))
                train_loss.backward()
                self.model_optimizer.step()

                print("Done batch!")


            ## test
            with torch.no_grad():
                save_idx = 0
                for batch_data_test in test_data_loader:
                    original_input = torch.clone(batch_data_test)
                    batch_data_test = batch_data_test.to(self.device)
                    angles = get_random_angles_array(batch_data_test.shape[0]).to(self.device)
                    for i in range(batch_data_test.shape[0]):
                        batch_data_test[i] = torchvision.transforms.functional.rotate(batch_data_test[i], float(angles[i])).to(self.device)

                    original_input_rotated = torch.clone(batch_data_test)

                    feat = self.model(batch_data_test).to(self.device)

                    print("test loss: ", criterion(feat, angles).data)

                    for i in range(batch_data_test.shape[0]):
                        batch_data_test[i] = torchvision.transforms.functional.rotate(batch_data_test[i], -1 * float(torch.argmax(feat[i]))).to(self.device)

                    test_examples_cpu = batch_data_test.cpu()

                    for index in range(int(len(test_examples_cpu)/10)):
                        img = test_examples_cpu[index].numpy()
                        orig_input = original_input[index].numpy()
                        orig_in_rotated = original_input_rotated[index].cpu().numpy()

                        plt.imsave(str("./test_output_rotation/" + str(epoch) + "_" + str(save_idx) + ".jpg"),
                                   np.concatenate((orig_input.transpose(1, 2, 0),
                                                   orig_in_rotated.transpose(1, 2, 0), img.transpose(1, 2, 0)), axis=1))

                        save_idx += 1

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), str("./Model_Weights/" + "gen_weights" + "_" + str(epoch)))

            print("Done epoch", epoch, "!")
