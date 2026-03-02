repo_path = 'libs/stylegan2-ada-pytorch-main/'
import numpy, cv2, os, dlib, subprocess, scipy, torch, sys, pickle
sys.path.append(repo_path)
import projector
import matplotlib.pyplot as plt
from PIL import Image


class LandmarksDetector:
    def __init__(self, predictor_model_path='libs/utils/shape_predictor_68_face_landmarks.dat'):
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
    def get_landmarks(self, image):
        print("Computing landmarks %s" %(image))
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)
        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except:
                print("Exception in get_landmarks()!")

landmarks_detector = LandmarksDetector();

def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        lm = numpy.array(face_landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = numpy.mean(lm_eye_left, axis=0)
        eye_right    = numpy.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - numpy.flipud(eye_to_mouth) * [-1, 1]
        x /= numpy.hypot(*x)
        x *= max(numpy.hypot(*eye_to_eye) * 2.0, numpy.hypot(*eye_to_mouth) * 1.8)
        x *= x_scale
        y = numpy.flipud(x) * [-y_scale, y_scale]
        c = eye_avg + eye_to_mouth * em_scale
        quad = numpy.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = numpy.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = Image.open(src_file).convert('RGBA').convert('RGB')

        # Shrink.
        shrink = int(numpy.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(numpy.rint(float(img.size[0]) / shrink)), int(numpy.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.Resampling.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(numpy.rint(qsize * 0.1)), 3)
        crop = (int(numpy.floor(min(quad[:,0]))), int(numpy.floor(min(quad[:,1]))), int(numpy.ceil(max(quad[:,0]))), int(numpy.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(numpy.floor(min(quad[:,0]))), int(numpy.floor(min(quad[:,1]))), int(numpy.ceil(max(quad[:,0]))), int(numpy.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = numpy.maximum(pad, int(numpy.rint(qsize * 0.3)))
            img = numpy.pad(numpy.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = numpy.ogrid[:h, :w, :1]
            mask = numpy.maximum(1.0 - numpy.minimum(numpy.float32(x) / pad[0], numpy.float32(w-1-x) / pad[2]), 1.0 - numpy.minimum(numpy.float32(y) / pad[1], numpy.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * numpy.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (numpy.median(img, axis=(0,1)) - img) * numpy.clip(mask, 0.0, 1.0)
            img = numpy.uint8(numpy.clip(numpy.rint(img), 0, 255))
            if alpha:
                mask = 1-numpy.clip(3.0 * mask, 0.0, 1.0)
                mask = numpy.uint8(numpy.clip(numpy.rint(mask*255), 0, 255))
                img = numpy.concatenate((img, mask), axis=2)
                img = Image.fromarray(img, 'RGBA')
            else:
                img = Image.fromarray(img, 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

        # Save aligned image.
        img.save(dst_file, 'PNG')

def AlignFace(File, TempFile):
	print("Alignining %s" %(File));
	face_landmarks = landmarks_detector.get_landmarks(File);
	for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(File), start = 1):
		image_align(File, TempFile, face_landmarks)

def MorphFace(Options): #Sun 07 Dec 2025 11:56:11 GMT 

	alpha = Options.Alpha;

	assert alpha > 0 and alpha < 1, "alpha not in [0,1]";
	assert os.path.exists(Options.Sb1), "File not found " + Options.Sb1;
	assert os.path.exists(Options.Sb2), "File not found " + Options.Sb2;
	
	aligned_face_path = "./temp_0.png";	
	AlignFace(Options.Sb1, aligned_face_path);
	Options.Sb1 = aligned_face_path;
	aligned_face_path = "./temp_1.png";
	AlignFace(Options.Sb2, aligned_face_path);
	Options.Sb2 = aligned_face_path;	

	face1 = Image.open(Options.Sb1)
	face2 = Image.open(Options.Sb2)

	face1_array = numpy.array(face1)
	face2_array = numpy.array(face2)

	face1_tensor = torch.tensor(face1_array, dtype=torch.float32)  # Change dtype as needed
	face2_tensor = torch.tensor(face2_array, dtype=torch.float32)

	# Set device
	if(torch.backends.mps.is_available()): # True
		print("MPS is available")
		device = torch.device("mps")
	else:
		device = torch.device("cpu")


	# Load the generator model from the pickle file
	with open('./MODELS/ffhq_res256.pkl', 'rb') as f:
		G = pickle.load(f)['G_ema'].to(device) 
    
	face1_tensor = face1_tensor.squeeze()
	face1_tensor = face1_tensor.permute(2, 0, 1)
	face1_tensor = torch.nn.functional.interpolate(face1_tensor.unsqueeze(0), size=(G.img_resolution, G.img_resolution), mode='bilinear', align_corners=False)
	face1_tensor = face1_tensor.squeeze(0)

	face2_tensor = face2_tensor.squeeze()
	face2_tensor = face2_tensor.permute(2, 0, 1)
	face2_tensor = torch.nn.functional.interpolate(face2_tensor.unsqueeze(0), size=(G.img_resolution, G.img_resolution), mode='bilinear', align_corners=False)
	face2_tensor = face2_tensor.squeeze(0)

	face1_tensor = face1_tensor.to(device)
	face2_tensor = face2_tensor.to(device)

	# Project the image
	projected_w_steps1 = projector.project(G, target=face1_tensor, num_steps=2, w_avg_samples = 1000, device = device, verbose=False);

	projected_w_steps2 = projector.project(G, target=face2_tensor, num_steps=2, w_avg_samples = 1000,   device=device, verbose=False)

	# check if the projected_w_steps1 and projected_w_steps2 are exactly the same
	(projected_w_steps1 == projected_w_steps2).all()


	w1 = projected_w_steps1[-1].unsqueeze(0)
	w2 = projected_w_steps2[-1].unsqueeze(0)

	img1 = G.synthesis(w1, noise_mode='const', force_fp32=True)

	sz_im = 255;

	img1 = (img1 + 1) * (sz_im/2)
	img1 = img1.permute(0, 2, 3, 1).clamp(0, sz_im).to(torch.uint8)[0].cpu().numpy()

	img1 = Image.fromarray(img1, 'RGB')
	img1.save("temp_1.png")

	img2 = G.synthesis(w2, noise_mode='const', force_fp32=True)

	img2 = (img2 + 1) * (sz_im/2)
	img2 = img2.permute(0, 2, 3, 1).clamp(0, sz_im).to(torch.uint8)[0].cpu().numpy()

	img2 = Image.fromarray(img2, 'RGB')
	img2.save("temp_2.png")

	# linear interpolation between w1 and w2
	num_interpolations = 10
	interpolations = torch.zeros((num_interpolations, w1.shape[1], w1.shape[2]), device=device)
	for i in range(num_interpolations):
		interpolations[i] = w1 + (w2 - w1) * i / (num_interpolations - 1)

	# Generate the images
	interpolated_images = G.synthesis(interpolations, noise_mode='const', force_fp32=True)

	interpolated_images.shape
	interpolated_images[5].shape

	interpolated_images = (interpolated_images + 1) * (sz_im/2)

	interpolated_images = interpolated_images.permute(0, 2, 3, 1).clamp(0, sz_im).to(torch.uint8).cpu().numpy()

	morph = interpolated_images[5];

	morph = Image.fromarray(morph, 'RGB')
	morph = morph.resize((1024, 1024))
	
	print("\nWritting morphed face : %s ... \n" %(Options.Morph));	
	morph.save(Options.Morph, "PNG");
	
	#CropFace(morphed_frame, img1, Options.Morph.replace(".png" , "_crop.png"), points)

	os.system("rm %s" %(Options.Sb1));
	os.system("rm %s" %(Options.Sb2));	
	
