import os

########## UAVID COLOURS
####### (0, 0, 0) --> black --> background clutter --> (0, 0, 0) #######
######## (0, 128, 0) --> green --> tree --> 22236 in ADE20K --> (4, 200, 3) #####
######  (64, 0, 128) --> purple --> moving car --> 14977 in ADE20K  --> (0, 102, 200) ###
#### (64, 64, 0) --> ochre --> humans --> 24420 in ADE20K --> (150, 5, 61) ###
### (128, 0, 0) --> red --> building ---> 18850 in ADE20K --> (180, 120, 120) ####
### (128, 64, 128) --> purple --> road --> 5396 in ADE20K  --> (140, 140, 140) ###
### (128, 128, 0) --> yellow --> low vegetation --> 4253 in ADE20K --> (4, 250, 7) #
### (192, 0, 192) --> neon purple --> static car --> 14977 in ADE20K --> (0, 102, 200) ##

colour_mappings = {
    (0, 0, 0): [0, 0, 0],
    (0, 128, 0): [4, 200, 3],
    (64, 0, 128): [0, 102, 200],
    (64, 64, 0): [150, 5, 61],
    (128, 0, 0): [180, 120, 120],
    (128, 64, 128): [140, 140, 140],
    (128, 128, 0): [4, 250, 7],
    (192, 0, 192): [0, 102, 200],
}


def apply_uavid_to_ade20k_map(colour_mappings, pix):

    for i in range(pix.shape[0]):
        for j in range(pix.shape[1]):
            pix[i][j] = colour_mappings[tuple(pix[i][j])]

    return pix

    train_path = DATASET_PATH / "uavid_val"


def create_new_segmentation_maps(path: Path):

    for seq in os.listdir(path):
        seq_full_path = str(path) + "/" + seq
        seq_labels = seq_full_path + "/Labels"
        new_labels = seq_full_path + "/ADE20K_Labels"
        os.makedirs(new_labels, exist_ok=True)

        for filename in os.listdir(seq_labels):
            filepath = os.path.join(seq_labels, filename)
            old_segmentation_map = Image.open(filepath)
            old_segmentation_map = numpy.array(old_segmentation_map)
            new_segmentation_map = apply_uavid_to_ade20k_map(
                colour_mappings, old_segmentation_map
            )
            new_image = Image.fromarray(new_segmentation_map)
            new_image.save(new_labels + "/" + filename)


create_new_segmentation_maps(train_path)

main_path = "/content/drive/MyDrive/FYP - video generation/uavid/uavid_train/"

seq_1 = "/content/drive/MyDrive/FYP - video generation/uavid/uavid_train/seq1"

seq_1_labels = seq_1 + "/Labels"
new_labels = seq_1 + "/ADE20K_Labels"
os.makedirs(new_labels, exist_ok=True)

for filename in os.listdir(seq_1_labels):
    filepath = os.path.join(seq_1_labels, filename)
    old_segmentation_map = Image.open(filepath)
    old_segmentation_map = numpy.array(old_segmentation_map)
    new_segmentation_map = apply_uavid_to_ade20k_map(
        colour_mappings, old_segmentation_map
    )
    new_image = Image.fromarray(new_segmentation_map)
    new_image.save(new_labels + "/" + filename)
