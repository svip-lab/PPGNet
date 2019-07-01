from data.line_graph import LineGraph
import os
from scipy import io
import numpy as np
from shutil import copyfile
from tqdm import trange

data_root = "/home/ziheng/YorkUrbanDB"
out_root = "/home/ziheng/YorkUrbanDB_new/test"


list_file = io.loadmat(os.path.join(data_root, "Manhattan_Image_DB_Names.mat"))
name_list = [e[0][0].strip("\\") for e in list_file["Manhattan_Image_DB_Names"]]
test_set = io.loadmat(os.path.join(data_root, "ECCV_TrainingAndTestImageNumbers.mat"))
test_set_id = test_set["testSetIndex"].flatten().tolist()
imgs = [os.path.join(data_root, name_list[i - 1], name_list[i - 1] + ".jpg") for i in test_set_id]
labels = [io.loadmat(os.path.join(data_root, name_list[i - 1], name_list[i - 1] + "LinesAndVP.mat")) for i in test_set_id]
lines = [np.float32(lab["lines"]).reshape((-1, 4)) for lab in labels]
maps = [np.uint8(lab["finalImg"]) for lab in labels]

os.makedirs(out_root, exist_ok=True)
max_juncs = 512
for i in trange(len(imgs)):
    img, line = imgs[i], lines[i]
    fname = os.path.basename(img)[:-4]
    hm = maps[i]
    lg = LineGraph(eps_junction=1., eps_line_deg=np.pi / 30, verbose=False)
    for x1, y1, x2, y2 in line:
        lg.add_junction((x1, y1))
        lg.add_junction((x2, y2))
    lg.freeze_junction()
    for x1, y1, x2, y2 in line:
        lg.add_line_seg((x1, y1), (x2, y2))
    lg.freeze_line_seg()
    max_juncs = max(lg.num_junctions, max_juncs)
    lg.save(os.path.join(out_root, fname + ".lg"))
    copyfile(img, os.path.join(out_root, fname + ".jpg"))
    # img = cv2.imread(img)
    # print(fname, flush=True)
    # cv2.imshow("line_", lg.line_map(img.shape[:2]))
    # cv2.imshow("line", hm)
    # cv2.waitKey()

print(max_juncs)
