import time
import numpy as np

    def is_mouth_open(self, keypoints):

        if len(keypoints) != 68:
            print("Not enough keypoints : ", np.array(keypoints).shape[0])
            return

        # print("Top Lower Lip: ", keypoints[66])
        # print("Bottom Lower Lip: ", keypoints[57])

        # print("Top Upper Lip: ", keypoints[51])
        # print("Bottom Upper Lip: ", keypoints[62])

        lipDist = np.sqrt(
            (keypoints[66][0] - keypoints[62][0]) ** 2
            + (keypoints[66][1] - keypoints[62][1]) ** 2
        )

        lipThickness = float(
            np.sqrt(
                (keypoints[51][0] - keypoints[62][0]) ** 2
                + (keypoints[51][1] - keypoints[62][1]) ** 2
            )
            / 2
        ) + float(
            np.sqrt(
                (keypoints[57][0] - keypoints[66][0]) ** 2
                + (keypoints[57][1] - keypoints[66][1]) ** 2
            )
            / 2
        )

        # print("lipDist: ",lipDist)
        # print("lipThickness: ",lipThickness)

        if lipDist >= 1.5 * lipThickness:
            return True, lipDist / lipThickness
        else:
            return False, lipDist / lipThickness

# TODO put other starter stuff here
