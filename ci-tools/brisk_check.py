"""Re-create Features2d_DescriptorExtractor_BRISK.regression: same image, stored keypoints, reference, Hamming rule."""
import struct, sys
import numpy as np
import cv2 as cv
data = sys.argv[1] + "/cv/features2d/"
img = cv.imread(data + "tsukuba.png")
fs = cv.FileStorage(data + "keypoints.xml.gz", cv.FILE_STORAGE_READ)
node = fs.getFirstTopLevelNode()
vals = np.array([node.at(i).real() for i in range(node.size())]).reshape(-1, 7)
kps = [cv.KeyPoint(float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), int(v[5]), int(v[6])) for v in vals]
with open(data + "descriptor_extractors/descriptor-brisk", "rb") as f:
    rows, cols, typ, size = struct.unpack("<4i", f.read(16))
    ref = np.frombuffer(f.read(size), np.uint8).reshape(rows, -1)[:, :cols]
make = getattr(cv, "BRISK_create", None) or cv.BRISK.create
_, desc = make().compute(img, kps)
if desc is None or desc.shape != ref.shape:
    print(f"{cv.__version__}: shape mismatch computed={None if desc is None else desc.shape} ref={ref.shape} kps={len(kps)}"); sys.exit()
dist = np.unpackbits(np.bitwise_xor(desc, ref), axis=1).sum(axis=1)
print(f"{cv.__version__}: rows={rows} exact={int((dist==0).sum())} max={int(dist.max())} over2={int((dist>2).sum())} ({100*(dist>2).mean():.1f}%)")
