import math
import torch
import time
import argparse
from model import DehazeModel
from val_dataset import dehaze_val_dataset
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image as imwrite
from tqdm import tqdm

import pdb
DEHAZE_ZEROPAD_TIMES = 16

def zeropad_tensor(tensor, times=32):
    B, C, H, W = tensor.shape
    Hnew = int(times * math.ceil(H / times))
    Wnew = int(times * math.ceil(W / times))
    temp = tensor.new_zeros(B, C, Hnew, Wnew)
    temp[:, :, 0:H, 0:W] = tensor
    return temp

def model_forward(model, input_tensor):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % DEHAZE_ZEROPAD_TIMES != 0 or W % DEHAZE_ZEROPAD_TIMES != 0:
        input_tensor = zeropad_tensor(input_tensor, times=DEHAZE_ZEROPAD_TIMES)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    return output_tensor[:, :, 0:H, 0:W]


# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description="RCAN-Dehaze-teacher")
parser.add_argument("--data_dir", type=str, default="")
parser.add_argument("--model_save_dir", type=str, default="output")
args = parser.parse_args()

val_dataset = os.path.join(args.data_dir, "NTIRE2021_Test_Hazy")

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join(args.model_save_dir, "")

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
model = DehazeModel()
print("model parameters:", sum(param.numel() for param in model.parameters()))

val_dataset = dehaze_val_dataset(val_dataset)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
# model= torch.nn.DataParallel(model, device_ids=device_ids)

# --- Load the network weight --- #
try:
    # module.tail1.1.bias
    weights = torch.load("models/best.pkl")
    model.load_state_dict({k.replace("module.", ""): v for k, v in weights.items()})
    # model.load_state_dict(weights)
    print("--- weight loaded ---")

except:
    print("--- no weight loaded ---")

# model = torch.jit.script(model)
model = model.to(device)
model.eval()

# --- Strat testing --- #
time_list = []

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_list = val_loader.dataset.list_test
#  val_loader.dataset.list_test -- ['31.png', '32.png', '33.png', '34.png', '35.png']

progress_bar = tqdm(total=len(file_list))
for i, hazy in enumerate(val_loader):
    progress_bar.update(1)

    # print(len(val_loader))
    hazy = hazy.to(device)

    start = time.time()

    # with torch.no_grad():
    #     img_tensor = model(hazy)
    img_tensor = model_forward(model, hazy)
    # hazy.size() -- [1, 3, 1200, 1600], hazy.min(), hazy.max() -- 0.2627, 0.9882
    # (Pdb) img_tensor.size() -- [1, 3, 1200, 1600]
    # (Pdb) img_tensor.min(), img_tensor.max() -- (0.0026, 0.9884)

    end = time.time()
    time_list.append((end - start))

    output_file = f"{output_dir}/{os.path.basename(file_list[i])}"
    imwrite(img_tensor, output_file)

time_cost = float(sum(time_list) / len(time_list))
print("running time per image: ", time_cost)
