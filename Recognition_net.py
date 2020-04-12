from __future__ import print_function
import argparse
import os

# Torch framework imports
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

# Image proccessing imports
import numpy as np
import cv2
from PIL import Image


#  RetinaFace model initialization imports
from ret_data import cfg_mnet, cfg_re50
from ret_models.retinaface import RetinaFace
from ret_src.ret_init import ret_load_model

#  RetinaFace landmarks decoding and Timer utility imports
from ret_layers.functions.prior_box import PriorBox
from ret_utils.nms.py_cpu_nms import py_cpu_nms
from ret_utils.box_utils import decode, decode_landm
from ret_utils.timer import Timer

#  CORAL model initialization import
from cor_src.cor_init import resnet34

#  Gender and Emotion model initialization imports
from keras.models import load_model
from keras.preprocessing import image as keras_image
from fa_utils.preprocessor import preprocess_input

parser = argparse.ArgumentParser(description='Recognition inputs')
parser.add_argument('--data_folder', default= r'./cor_LSTM_test' ,type=str, help='Dir to analyze')
parser.add_argument('--save_folder', default= './cor_RESULTS_comparison/Author_weights/',type=str, help='Dir to save results')
parser.add_argument('-m', '--RetinaFace_arch', default='MobileNet', type=str, help='MobileNet or ResNet')
parser.add_argument('--CORAL_weights', default= 'MORPH' , type=str, help='AFAD,CACD,MORPH,UTK')
parser.add_argument('--device', default='cuda', type=str, help='Proccesing unit CPU = cpu or GPU = cuda')
args = parser.parse_args()

#### Misc init
torch.set_grad_enabled(False)
torch.cuda.empty_cache()
font = cv2.FONT_HERSHEY_SIMPLEX
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
image_count = 0
_t = {'RET_forward_pass': Timer(), 'RET_misc': Timer(), 'COR_forward_pass': Timer(), 'COR_misc': Timer(), 'FA_gender_forward_pass': Timer(), 'FA_emotion_forward_pass': Timer(), 'Image_forward_pass': Timer()}


if __name__ == '__main__':

    if args.save_folder == None or args.data_folder == None:
       print('Incorrectly entered path to data or save folder')
       raise FileNotFoundError()
    save_directory = args.save_folder
    test_directory = args.data_folder
    file_count = sum([len(files) for r, d, files in os.walk(test_directory)])
    device = torch.device(args.device)

    ##### RET net and model

    if args.RetinaFace_arch == 'MobileNet':
       cfg = cfg_mnet
       RetinaFace_net = RetinaFace(cfg=cfg, phase='test')
       RetinaFace_net = ret_load_model(RetinaFace_net, 'ret_weights/mobilenet0.25_Final.pth', device)
    else:
        cfg = cfg_re50
        RetinaFace_net = RetinaFace(cfg=cfg, phase='test')
        RetinaFace_net = ret_load_model(RetinaFace_net, 'ret_weights/Resnet50_Final.pth', device)
    RetinaFace_net.eval()
    RetinaFace_net = RetinaFace_net.to(device)

    ##### COR net and model
    # 1- AFAD Age range :15-40
    # 2- CACD Age range :14-62
    # 3- MORPH Age range :16-70
    # 4- UTK Age range :21-60

    coral_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.CenterCrop((120, 120)),
                                          transforms.ToTensor()])

    coral_weight_dict ={'AFAD': [26,15,'cor_weights/afad-coral_seed0_imp0/model.pt'], 'CACD': [49,14,'cor_weights/cacd-coral_seed0_imp0/model.pt'],
                        'MORPH': [55,16,'cor_weights/morph-coral_seed0_imp0/model.pt'], 'UTK': [40,21,'cor_weights/utk-coral_seed0_imp0/model.pt']}

    coral_weight_input = coral_weight_dict[args.CORAL_weights]
    NUM_CLASSES = coral_weight_input[0]
    dataset_lowest_age = coral_weight_input[1]
    weights_path = coral_weight_input[2]
    CORAL_net = resnet34(NUM_CLASSES, False)
    print('CORAL network status: Loading weights from {}'.format(weights_path))
    CORAL_net.load_state_dict(torch.load(weights_path,map_location=device))
    print('CORAL network status: Loaded')
    CORAL_net.to(device)
    CORAL_net.eval()

    ######## FA load

    emotion_model_path = './fa_trained_models/gpu_mini_XCEPTION.63-0.64.hdf5'
    gender_model_path = './fa_trained_models/gender_models/simple_CNN.81-0.96.hdf5'

    gender_labels = {0: 'woman', 1: 'man'}
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

    print('FA gender network status: Loading weights from {}'.format(gender_model_path))
    FA_gender_net = load_model(gender_model_path, compile=False)
    print('FA gender network status: Loaded')
    print('FA emotion network status: Loading weights from {}'.format(emotion_model_path))
    FA_emotion_net = load_model(emotion_model_path, compile=False)
    print('FA emotion network status: Loaded')

    gender_target_size = FA_gender_net.input_shape[1:3]
    emotion_target_size = FA_emotion_net .input_shape[1:3]

    for (dirpath, dirnames, filenames) in os.walk(test_directory):
        for filename in filenames:
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".tiff"):

                _t['Image_forward_pass'].tic()

                # Image read
                image_count += 1
                face_counter = 0
                image_path = os.path.join(dirpath,filename)
                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = np.float32(img_raw)

                if img.shape[2] < 3: break  # ignore gray images

                # Testing scale
                target_size = 1600
                max_size = 2150
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                resize = float(target_size) / float(im_size_min)
                # Prevent bigger axis from being more than max_size:
                if np.round(resize * im_size_max) > max_size:
                    resize = float(max_size) / float(im_size_max)
                if True:
                    resize = 1

                if resize != 1:
                    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                _t['RET_forward_pass'].tic()
                loc, conf, landms = RetinaFace_net(img)  # forward pass
                _t['RET_forward_pass'].toc()

                _t['RET_misc'].tic()

                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2]])
                scale1 = scale1.to(device)
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > 0.02)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1]
                # order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]

                # Non Maximum Suppression
                bboxs = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(bboxs, 0.4)
                bboxs = bboxs[keep, :]
                landms = landms[keep]

                bboxs = np.concatenate((bboxs, landms), axis=1)

                _t['RET_misc'].toc()

                ##################################
                ######  Facial analysis part #####
                ##################################

                save_name = save_directory + '\\' + filename[:-4] + ".txt"
                dirname   = os.path.dirname(save_name)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                file_name = os.path.basename(save_name)[:-4] + "\n"

                u_image = Image.open(image_path)
                if u_image.mode != 'RGB':
                   break

                bboxs_num = "Number of proposed BB: " + str(len(bboxs)) + "\n\n"
                with open(save_name, "w") as fd:
                    fd.write(file_name)
                    fd.write(bboxs_num)

                    for box in bboxs:

                        if box[4] < 0.97:
                           continue
                        face_counter+=1
                        x = int(box[0])
                        y = int(box[1])
                        w = int(box[2]) - int(box[0])
                        h = int(box[3]) - int(box[1])
                        confidence = str(box[4])


                        #####  CORAL inference #####

                        face_image = u_image.crop((box[0], box[1], box[2], box[3]))

                        image = coral_transform(face_image)
                        image = image.to(device)
                        image = image.unsqueeze(0)

                        _t['COR_forward_pass'].tic()
                        logits, probas = CORAL_net(image)  # forward pass
                        _t['COR_forward_pass'].toc()

                        predict_levels = probas > 0.5
                        predicted_label = torch.sum(predict_levels, dim=1)
                        age_text = str(predicted_label.item() + dataset_lowest_age)

                        ##### Gender and Emotion inference #####

                        rgb_image =  keras_image.load_img(image_path, color_mode="rgb")
                        rgb_image =  keras_image.img_to_array(rgb_image)

                        gray_image = keras_image.load_img(image_path, color_mode="grayscale")
                        gray_image = keras_image.img_to_array(gray_image)

                        gray_image = np.squeeze(gray_image)
                        gray_image = gray_image.astype('uint8')

                        ctrl_h = 3
                        ctrl_w = 4
                        x_1 = int(box[1]) - int(h / ctrl_h)
                        x_2 = int(box[3]) + int(h / ctrl_h)
                        y_1 = int(box[0]) - int(w / ctrl_w)
                        y_2 = int(box[2]) + int(w / ctrl_w)
                        if x_1 < 0 : x_1 = 0
                        if y_1 < 0:  y_1 = 0

                        rgb_face = rgb_image[x_1:x_2,y_1:y_2]
                        gray_face = gray_image[x_1:x_2,y_1:y_2]

                        rgb_face = cv2.resize(rgb_face, (gender_target_size))
                        gray_face = cv2.resize(gray_face, (emotion_target_size))

                        rgb_face = preprocess_input(rgb_face, False)
                        rgb_face = np.expand_dims(rgb_face, 0)
                        _t['FA_gender_forward_pass'].tic()
                        gender_prediction = FA_gender_net .predict(rgb_face)  # forward pass
                        _t['FA_gender_forward_pass'].toc()
                        gender_label_arg = np.argmax(gender_prediction)
                        gender_text = gender_labels[gender_label_arg]

                        gray_face = preprocess_input(gray_face, True)
                        gray_face = np.expand_dims(gray_face, 0)
                        gray_face = np.expand_dims(gray_face, -1)
                        _t['FA_emotion_forward_pass'].tic()
                        emotion_label_arg = np.argmax(FA_emotion_net .predict(gray_face))  # forward pass
                        _t['FA_emotion_forward_pass'].toc()
                        emotion_text = emotion_labels[emotion_label_arg]

                        if gender_text == gender_labels[0]:
                            color = (0, 0, 255)
                        else:
                            color = (255, 0, 0)

                        ##### Saving predictions #####

                        # Write bounding box description and predictions into .txt file

                        line =  ("x: "+str(x)+ " \n"
                               +"y: "+str(y)+ " \n"
                               +"w: "+str(w)+ " \n"
                               +"w: "+str(h)+ " \n"
                               +"probability: "+ confidence + " \n"
                               +"age: " + age_text + " \n"
                               +"gender: " + gender_text + " \n"
                               +"emotion: " + emotion_text + " \n"
                               +"left eye - x: "+str(box[5])+" ,y: "+str(box[6])+" \n"
                               +"right eye - x: "+str(box[7])+" ,y: "+str(box[8])+" \n"
                               +"nose tip - x: "+str(box[9])+" ,y: "+str(box[10])+" \n"
                               +"mouth left corner - x: "+str(box[11])+" ,y: "+str(box[12])+" \n"
                               +"mouth right corner - x: "+str(box[13])+" ,y: "+str(box[14])+" \n\n""\n")
                        fd.write(line)

                        #  Draw predictions onto image

                        box = list(map(int, box))
                        cv2.rectangle(img_raw, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 12)
                        cv2.rectangle(img_raw, (box[0], box[1]), (box[2], box[3]), color, 6)
                        # cv2.rectangle(img_raw, (y_1, x_1), (y_2, x_2), (255, 255, 65), 6)

                        cv2.circle(img_raw, (box[5], box[6]), 1, (0, 0, 255), 3)
                        cv2.circle(img_raw, (box[7], box[8]), 1, (0, 255, 255), 3)
                        cv2.circle(img_raw, (box[9], box[10]), 1, (255, 0, 255), 3)
                        cv2.circle(img_raw, (box[11], box[12]), 1, (0, 255, 0), 3)
                        cv2.circle(img_raw, (box[13], box[14]), 1, (255, 0, 0), 3)

                        cx = box[0] + int(im_width*0.03)
                        cy1 = box[1] - int(im_height*0.008)
                        cy2 = box[1] + int(h*1.2)
                        cv2.putText(img_raw, age_text, (cx, cy2),
                                    font, 1.5, (0, 0, 0),5)
                        cv2.putText(img_raw, age_text,  (cx, cy2),
                                    font, 1.5, (255, 255, 255),4)
                        cv2.putText(img_raw, gender_text, (cx, cy2+40),
                                    font, 1.5, (0, 0, 0), 5)
                        cv2.putText(img_raw, gender_text, (cx, cy2+40),
                                    font, 1.5, (255, 255, 255), 4)
                        cv2.putText(img_raw, emotion_text, (cx, cy2+80),
                                    font, 1.5, (0, 0, 0), 5)
                        cv2.putText(img_raw, emotion_text, (cx, cy2+80),
                                    font, 1.5, (255, 255, 255), 4)


                        _t['Image_forward_pass'].toc()

                        #  Print status of currently ongoing analysis of folder

                        print('filename: '+filename+' \n'
                              'image_counter: {:d}/{:d} \n'
                              'face_counter: {:d} \n'
                              'RetinaFace - forward_pass_time: {:.4f}s \n'
                              'CORAL - forward_pass_time: {:.4f}s \n'
                              'Facial analysis gender - forward_pass_time: {:.4f}s \n'
                              'Facial analysis emotion - forward_pass_time: {:.4f}s \n'
                              'Image average - forward_pass_time: {:.4f}s \n\n'.format(image_count, file_count,
                                                                          face_counter,
                                                                         _t['RET_forward_pass'].average_time,
                                                                         _t['COR_forward_pass'].average_time,
                                                                         _t['FA_gender_forward_pass'].average_time,
                                                                         _t['FA_emotion_forward_pass'].average_time,
                                                                         _t['Image_forward_pass'].average_time))

                # save image
                name = save_directory + '\\'+ str(image_count) + ".jpg"
                cv2.imwrite(name, img_raw)





