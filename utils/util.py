import numpy as np
import torch
import random
from PIL import ImageDraw, Image

pairs_tr = [(0, '049'), (1, '128'), (2, '116'), (3, '085'), (4, '232'), (5, '342'), (6, '340'), (7, '025'), (8, '043'), (9, '644'), (10, '123'), (11, '343'), (12, '000'), (13, '158'), (14, '339'), (15, '151'), (16, '111'), (17, '013'), (18, '018'), (19, '127'), (20, '118'), (21, '332'), (22, '211'), (23, '090'), (24, '126'), (25, '060'), (26, '133'), (27, '247'), (28, '153'), (29, '338'), (30, '019'), (31, '333'), (32, '337'), (33, '259'), (34, '042'), (35, '233'), (36, '156'), (37, '136'),
        (38, '246'), (39, '663'), (40, '063'), (41, '135'), (42, '328'), (43, '635'), (44, '007'), (45, '100'), (46, '217'), (47, '227'), (48, '242'), (49, '239'), (50, '670'), (51, '640'), (52, '012'), (53, '125'), (54, '216'), (55, '254'), (56, '054'), (57, '152'), (58, '026'), (59, '223'), (60, '109'), (61, '334'), (62, '139'), (63, '081'), (64, '021'), (65, '066'), (66, '008'), (67, '105'), (68, '113'), (69, '265'), (70, '107'), (71, '336'), (72, '083'), (73, '330'), (74, '112'), (75,
            '046'), (76, '047'), (77, '245'), (78, '062'), (79, '149'), (80, '647'), (81, '016'), (82, '215'), (83, '654'), (84, '033'), (85, '150'), (86, '028'), (87, '273'), (88, '110'), (89, '059'), (90, '142'), (91, '275'), (92, '077'), (93, '064'), (94, '155'), (95, '671'), (96, '044'), (97, '117'), (98, '003'), (99, '272'), (100, '341'), (101, '058'), (102, '075'), (103, '079'), (104, '161'), (105, '214'), (106, '240'), (107, '130'), (108, '088'), (109, '146'), (110, '005'), (111,
                '257'), (112, '073'), (113, '241'), (114, '017'), (115, '134'), (116, '163'), (117, '159'), (118, '160'), (119, '243'), (120, '099'), (121, '002'), (122, '256'), (123, '164'), (124, '068'), (125, '144'), (126, '071'), (127, '137'), (128, '093'), (129, '335'), (130, '228'), (131, '014'), (132, '065'), (133, '636'), (134, '038'), (135, '087'), (136, '080'), (137, '230'), (138, '260'), (139, '069'), (140, '031'), (141, '220'), (142, '086'), (143, '010'), (144, '258'), (145,
                    '096'), (146, '061'), (147, '092'), (148, '004'), (149, '050'), (150, '106'), (151, '253'), (152, '143'), (153, '084'), (154, '006'), (155, '027'), (156, '344'), (157, '166'), (158, '162'), (159, '040'), (160, '009'), (161, '097'), (162, '132'), (163, '131'), (164, '056'), (165, '053'), (166, '218'), (167, '660'), (168, '037'), (169, '095'), (170, '252'), (171, '261'), (172, '268'), (173, '643'), (174, '224'), (175, '115'), (176, '015'), (177, '108'), (178, '124'),
                (179, '274'), (180, '234'), (181, '664'), (182, '219'), (183, '045'), (184, '659'), (185, '248'), (186, '231'), (187, '154'), (188, '039'), (189, '104'), (190, '082'), (191, '235'), (192, '034'), (193, '326'), (194, '221'), (195, '666'), (196, '048'), (197, '091'), (198, '024'), (199, '094'), (200, '102'), (201, '036'), (202, '237'), (203, '023'), (204, '035'), (205, '638'), (206, '329'), (207, '212'), (208, '120'), (209, '263'), (210, '667'), (211, '011'), (212, '213'),
                (213, '114'), (214, '641'), (215, '238'), (216, '051'), (217, '651'), (218, '236'), (219, '122'), (220, '229'), (221, '030'), (222, '129'), (223, '140'), (224, '074'), (225, '225'), (226, '637'), (227, '648'), (228, '157'), (229, '001'), (230, '165'), (231, '662'), (232, '119'), (233, '089'), (234, '645'), (235, '255'), (236, '658'), (237, '649'), (238, '639'), (239, '250'), (240, '041'), (241, '070'), (242, '103'), (243, '331'), (244, '052'), (245, '262'), (246, '650'),
                (247, '244'), (248, '652'), (249, '020'), (250, '147'), (251, '661'), (252, '121'), (253, '078'), (254, '138'), (255, '665'), (256, '145'), (257, '148'), (258, '264'), (259, '668'), (260, '141'), (261, '267'), (262, '167'), (263, '067'), (264, '222'), (265, '249'), (266, '251'), (267, '270'), (268, '642'), (269, '266'), (270, '655'), (271, '669'), (272, '327'), (273, '653'), (274, '076'), (275, '055'), (276, '226'), (277, '269'), (278, '029'), (279, '032'), (280, '072'),
                (281, '022'), (282, '098'), (283, '559'), (284, '582'), (285, '551'), (286, '529'), (287, '611'), (288, '511'), (289, '515'), (290, '197'), (291, '540'), (292, '606'), (293, '527'), (294, '624'), (295, '519'), (296, '289'), (297, '548'), (298, '320'), (299, '323'), (300, '531'), (301, '202'), (302, '184'), (303, '178'), (304, '613'), (305, '627'), (306, '176'), (307, '574'), (308, '621'), (309, '193'), (310, '586'), (311, '324'), (312, '521'), (313, '566'), (314, '595'),
                (315, '533'), (316, '182'), (317, '604'), (318, '612'), (319, '632'), (320, '174'), (321, '541'), (322, '290'), (323, '625'), (324, '210'), (325, '549'), (326, '567'), (327, '575'), (328, '576'), (329, '310'), (330, '192'), (331, '278'), (332, '544'), (333, '605'), (334, '626'), (335, '628'), (336, '301'), (337, '317'), (338, '171'), 
# ],
                 (339, '477'), (340, '469'), (341, '425'), (342, '355'), (343, '354'), (344, '406'), (345, '386'), (346, '473'), (347, '478'), (348, '294'), (349, '499'), (350, '353'), (351, '461'), (352, '488'), (353, '389'), (354, '423'), (355, '436'), (356, '446'), (357, '372'), (358, '428'), (359, '351'), (360, '456'), (361, '392'), (362, '375'), (363, '455'), (364, '410'), (365, '482'), (366, '434'), (367, '368'), (368, '435'), (369, '366'), (370, '487'), (371, '422'), (372, '360'), (373, '501'), 
                (374, '404'), (375, '347'), (376, '397'), (377, '387'), (378, '373'), (379, '494'), (380, '385'), (381, '405'), (382, '490'), (383, '418'), (384, '467'), (385, '480'), (386, '384'), (387, '497'), (388, '414'), (389, '462'), (390, '475'), (391, '399'), (392, '505'), (393, '492'), (394, '503'), (395, '407'), (396, '442'), (397, '396'), (398, '443'), (399, '345'), (400, '411'), (401, '445'), (402, '371'), (403, '453'), (404, '463'), (405, '426'), (406, '349'), (407, '391'), (408, '474'), 
                (409, '412'), (410, '495'), (411, '362'), (412, '359'), (413, '363'), (414, '481'), (415, '352'), (416, '483'), (417, '424'), (418, '493'), (419, '468'), (420, '393'), (421, '394'), (422, '452'), (423, '433'), (424, '504'), (425, '383'), (426, '408'), (427, '454'), (428, '380'), (429, '429'), (430, '470'), (431, '401'), (432, '427'), (433, '400'), (434, '421'), (435, '419'), (436, '350'), (437, '485'), (438, '379'), (439, '348'), (440, '420'), (441, '365'), (442, '367'), (443, '450'), 
                (444, '369'), (445, '471'), (446, '451'), (447, '398'), (448, '476'), (449, '413'), (450, '496'), (451, '346'), (452, '388'), (453, '458'), (454, '357'), (455, '465'), (456, '502'), (457, '356'), (458, '364'), (459, '441'), (460, '500'), (461, '390'), (462, '489'), (463, '377'), (464, '479'), (465, '449'), (466, '402'), (467, '484'), (468, '491'), (469, '409'), (470, '464'), (471, '459'), (472, '466'), (473, '376'), (474, '361'), (475, '448'), (476, '382'), (477, '378'), (478, '431'), 
                (479, '486'), (480, '444'), (481, '457'), (482, '440'), (483, '403'), (484, '472'), (485, '415'), (486, '395'), (487, '430'), (488, '447'), (489, '460'), (490, '432'), (491, '416'), (492, '370'), (493, '439'), (494, '498'), (495, '417')]
label2wid_tr = {k:v for k, v in pairs_tr}
wid2label_tr = {v:k for k, v in pairs_tr}

'''
description: Normalize the xy-coordinates into a standard interval.
Refer to "Drawing and Recognizing Chinese Characters with Recurrent Neural Network".
'''
def normalize_xys(xys):
    stroken_state = np.cumsum(np.concatenate((np.array([0]), xys[:, -2]))[:-1])
    px_sum = py_sum = len_sum = 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_px, temp_py = temp_len * (xy_1 + xy) / 2
            px_sum += temp_px
            py_sum += temp_py
            len_sum += temp_len
    if len_sum==0:
        raise Exception("Broken online characters")
    else:
        pass
    
    mux, muy = px_sum / len_sum, py_sum / len_sum
    dx_sum, dy_sum = 0, 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_dx = temp_len * (
                    np.power(xy_1[0] - mux, 2) + np.power(xy[0] - mux, 2) + (xy_1[0] - mux) * (xy[0] - mux)) / 3
            temp_dy = temp_len * (
                    np.power(xy_1[1] - muy, 2) + np.power(xy[1] - muy, 2) + (xy_1[1] - muy) * (xy[1] - muy)) / 3
            dx_sum += temp_dx
            dy_sum += temp_dy
    sigma = np.sqrt(dx_sum / len_sum)
    if sigma == 0:
        sigma = np.sqrt(dy_sum / len_sum)
    xys[:, 0], xys[:, 1] = (xys[:, 0] - mux) / sigma, (xys[:, 1] - muy) / sigma
    return xys

'''
description: Rendering offline character images by connecting coordinate points
'''
def coords_render(coordinates, split, width, height, thickness, board=5):
    canvas_w = width  
    canvas_h = height  
    board_w = board  
    board_h = board
    # preprocess canvas size
    p_canvas_w = canvas_w - 2*board_w
    p_canvas_h = canvas_h - 2*board_h

    # find original character size to fit with canvas
    min_x = 635535
    min_y = 635535
    max_x = -1
    max_y = -1
    
    coordinates[:, 0] = np.cumsum(coordinates[:, 0])
    coordinates[:, 1] = np.cumsum(coordinates[:, 1])
    if split:
        ids = np.where(coordinates[:, -1] == 1)[0] 
        if len(ids) < 1:  ### if not exist [0, 0, 1]
            ids = np.where(coordinates[:, 3] == 1)[0] + 1
            if len(ids) < 1: ### if not exist [0, 1, 0]
                ids = np.array([len(coordinates)])
                xys_split = np.split(coordinates, ids, axis=0)[:-1] # remove the blank list
            else:
                xys_split = np.split(coordinates, ids, axis=0)
        else:  ### if exist [0, 0, 1]
            remove_end = np.split(coordinates, ids, axis=0)[0]
            ids = np.where(remove_end[:, 3] == 1)[0] + 1 ### break in [0, 1, 0]
            xys_split = np.split(remove_end, ids, axis=0)
    else:
        pass
    for stroke in xys_split:
        for (x, y) in stroke[:, :2].reshape((-1, 2)):
            min_x = min(x, min_x)
            max_x = max(x, max_x)
            min_y = min(y, min_y)
            max_y = max(y, max_y)
    original_size = max(max_x-min_x, max_y-min_y)
    canvas = Image.new(mode='L', size=(canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        xys = np.stack([xs, ys], axis=-1).reshape(-1)
        xys[::2] = (xys[::2]-min_x) / original_size * p_canvas_w + board_w 
        xys[1::2] = (xys[1::2] - min_y) / original_size * p_canvas_h + board_h
        xys = np.round(xys)
        draw.line(xys.tolist(), fill=0, width=thickness)
    return canvas

# fix random seeds for reproducibility
def fix_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.manual_seed(random_seed)

### model loads specific parameters (i.e., par) from pretrained_model 
def load_specific_dict(model, pretrained_model, par):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_model)
    if par in list(pretrained_dict.keys())[0]:
        count = len(par) + 1
        pretrained_dict = {k[count:]: v for k, v in pretrained_dict.items() if k[count:] in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if len(pretrained_dict) > 0:
        model_dict.update(pretrained_dict)
    else:
        return ValueError
    return model_dict


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


'''
description: convert the np version of coordinates to the list counterpart
'''
def dxdynp_to_list(coordinates):
    ids = np.where(coordinates[:, -1] == 1)[0]
    length = coordinates[:, 2:4].sum()
    if len(ids) < 1:  ### if not exist [0, 0, 1]
        ids = np.where(coordinates[:, 3] == 1)[0] + 1
        if len(ids) < 1: ### if not exist [0, 1, 0]
            ids = np.array([len(coordinates)])
            xys_split = np.split(coordinates, ids, axis=0)[:-1] # remove the blank list
        else:
            xys_split = np.split(coordinates, ids, axis=0)
    else:  ### if exist [0, 0, 1]
        remove_end = np.split(coordinates, ids, axis=0)[0]
        ids = np.where(remove_end[:, 3] == 1)[0] + 1 ### break in [0, 1, 0]
        xys_split = np.split(remove_end, ids, axis=0)[:-1] # split from the remove_end
    
    coord_list = []
    for stroke in xys_split:
        xs, ys = stroke[:, 0], stroke[:, 1]
        if len(xs) > 0:
            xys = np.stack([xs, ys], axis=-1).reshape(-1)
            coord_list.append(xys)
        else:
            pass
    return coord_list, length

'''
description: 
    [x, y] --> [x, y, p1, p2, p3]
    see 'A NEURAL REPRESENTATION OF SKETCH DRAWINGS' for more details
'''
def corrds2xys(coordinates):
    new_strokes = []
    for stroke in coordinates:
        for (x, y) in np.array(stroke).reshape((-1, 2)):
            p = np.array([x, y, 1, 0, 0], np.float32)
            new_strokes.append(p)
        try:   
            new_strokes[-1][2:] = [0, 1, 0]  # set the end of a stroke
        except IndexError:
            print(stroke)
            return None
    new_strokes = np.stack(new_strokes, axis=0)
    return new_strokes