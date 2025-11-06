import cv2
import numpy as np
from PIL import Image
import glob
import random


def save(img, name="out.jpg"):
    im = Image.fromarray(img)  # (img * 255).astype(np.uint8)
    im.save(name)


class FeatureDescriptor:
    def __init__(self, x, y, descriptor):
        self.x = x
        self.y = y
        self.descriptor = descriptor

    def compute_similarity(self, other):
        # euclidean distance I guess
        return np.linalg.norm(self.descriptor - other.descriptor)

    def get_closest_match(self, others):
        best_match = None
        best_distance = float("inf")
        for other in others:
            dist = self.compute_similarity(other)
            if dist < best_distance:
                best_distance = dist
                best_match = other
        return best_match

    def get_closest_xy_match(self, others):
        best_match = None
        best_distance = float("inf")
        for other in others:
            dist = np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
            if dist < best_distance:
                best_distance = dist
                best_match = other
        return best_match


def descriptor(img, x, y, size=16):
    patch = img[y - size // 2 : y + size // 2, x - size // 2 : x + size // 2]

    # ??? do something so it's invariant to rotation
    # rotate it based on something. Gradient?

    gy, gx = np.gradient(patch)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    # pretty sure SIFT does something similar to this
    hist, bin_edges = np.histogram(
        angle, bins=36, range=(-np.pi, np.pi), weights=magnitude
    )
    dominant_angle = bin_edges[np.argmax(hist)]

    M = cv2.getRotationMatrix2D((size // 2, size // 2), dominant_angle * 180 / np.pi, 1)
    patch = cv2.warpAffine(patch, M, (size, size))

    # NOT scale-invariant but hopefully doesn't really matter on a small enough patch size. Fine tune size to get enough detail without noticing scale changes

    desc = patch.flatten()
    return desc


def descriptor_SIFT(
    img, x, y
):  # ai-generated function (gpt-5). Used for test PoC only, I hope to implement my own descriptor
    sift = cv2.SIFT_create()
    x = int(x)
    y = int(y)
    keypoint = cv2.KeyPoint(x, y, 1)
    _, descriptor = sift.compute(img, [keypoint])
    return descriptor[0]


def obtain_featuredescriptors(path):
    img_orig = np.array(Image.open(path), dtype=np.uint8)

    print(img_orig.shape)
    img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
    img = (img / np.max(img) * 255).astype(np.uint8)
    img = cv2.equalizeHist(
        img
    )  # attempt to make features invariant to lighting changes
    img = cv2.GaussianBlur(img, (5, 5), 0)
    features = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
    features = np.int32(features)
    print(
        "Magically obtained features found with cv2.goodFeaturesToTrack. Length: ",
        len(features),
    )
    drawing_img = img_orig.copy()
    for f in features:
        x, y = f.ravel()
        cv2.circle(drawing_img, (x, y), 10, (0, 255, 0), -1)
    save(drawing_img, "keypoints.png")

    features = np.int32(features)
    descriptors = np.array([descriptor_SIFT(img, f[0][0], f[0][1]) for f in features])

    print("Computed descriptors. Shape: ", descriptors.shape)

    feature_descriptors = [
        FeatureDescriptor(f[0][0], f[0][1], desc)
        for f, desc in zip(features, descriptors)
    ]

    return feature_descriptors


def match_next_image(descriptors1, descriptors2):
    # RANSAC
    # get four random points
    # compute homography
    # get whole score
    # repeat
    def attempt_get_homography():
        pts1 = random.choices(descriptors1, k=4)
        pts2 = [p.get_closest_match(descriptors2) for p in pts1]

        # compute homography
        H, _ = cv2.findHomography(
            np.array([[p.x, p.y] for p in pts1]), np.array([[p.x, p.y] for p in pts2])
        )
        if H is None:
            print("DING DING DING H IS NONE")
            return np.eye(3), float("inf")

        # get whole score
        score = 0
        for p in descriptors1:
            transformed = np.dot(H, np.array([p.x, p.y, 1]))
            transformed /= transformed[2]

            newp = FeatureDescriptor(transformed[0], transformed[1], p.descriptor)

            closest = newp.get_closest_xy_match(descriptors2)
            sim = newp.compute_similarity(closest)
            score += sim

        return H, score

    # repeat
    best_H = None
    best_score = float("inf")
    for _ in range(1000):
        H, score = attempt_get_homography()
        if score < best_score:
            best_score = score
            best_H = H
    print("Best score: ", best_score)
    return best_H


def match_next_image_RANSAC_builtin(descriptors1, descriptors2):
    # RANSAC
    # get four random points
    # compute homography
    # get whole score
    # repeat

    # compute homography
    H, _ = cv2.findHomography(
        np.array([[p.x, p.y] for p in descriptors1]),
        np.array(
            [
                [p.x, p.y]
                for p in [dp1.get_closest_match(descriptors2) for dp1 in descriptors1]
            ]
        ),
        cv2.RANSAC,
    )

    if H is None:
        print("DING DING DING H IS NONE")
        return np.eye(3), float("inf")

    H = np.array(H)

    # get whole score
    score = 0
    for p in descriptors1:
        transformed = np.dot(H, np.array([p.x, p.y, 1]))
        transformed /= transformed[2]

        closest = p.get_closest_xy_match(descriptors2)
        sim = p.compute_similarity(closest)
        score += sim

    print("RANSAC Best score: ", score)

    return H


def prepare(img):
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = (img / np.max(img) * 255).astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


if __name__ == "__main__":
    dt1 = descriptor_SIFT(prepare(Image.open("sample_imgs/test1.jpg")), 8, 8)
    dt2 = descriptor_SIFT(prepare(Image.open("sample_imgs/test2.jpg")), 8, 8)
    dt3 = descriptor_SIFT(prepare(Image.open("sample_imgs/test3_different.jpg")), 8, 8)

    print("Descriptor distance test for same image: ", np.linalg.norm(dt1 - dt2))
    print("Descriptor distance test for different image: ", np.linalg.norm(dt1 - dt3))

    d1 = obtain_featuredescriptors("images/1.jpg")
    d2 = obtain_featuredescriptors("images/2.jpg")
    H = match_next_image_RANSAC_builtin(d1, d2)
    print("Estimated homography: ")
    print(H)

    # apply H to d2
    img2 = np.array(Image.open("images/2.jpg"), dtype=np.uint8)
    img2_warped = cv2.warpPerspective(img2, H, (img2.shape[1] * 2, img2.shape[0] * 2))
    save(img2_warped, "warped.jpg")
