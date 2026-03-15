import numpy, cv2, os, dlib, subprocess, scipy, dlib
from PIL import Image

class LandmarksDetector:
    def __init__(self, predictor_model_path='libs/utils/shape_predictor_68_face_landmarks.dat'):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
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

class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass


def rect_contains(rect, point):

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Write the delaunay triangles into a file
def draw_delaunay(f_w, f_h, subdiv, dictionary1):
    list4 = []
    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)
    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            list4.append((int(dictionary1[pt1]), int(dictionary1[pt2]), int(dictionary1[pt3])));

    dictionary1 = {}
    return list4

def make_delaunay(f_w, f_h, theList, img1, img2):

    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary. 
    theList = theList.tolist()
    points = [(int(x[0]),int(x[1])) for x in theList]
    dictionary = {x[0]:x[1] for x in list(zip(points, range(76)))}
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)
   
    # Return the list.
    return list4
    
def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]

def crop_image(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        scale0 = size1[0]/size2[0]
        scale1 = size1[1]/size2[1]
        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        scale0 = size2[0]/size1[0]
        scale1 = size2[1]/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]
    
    else:
        return [img1[:,diff1:avg1],img2[-diff0:avg0,:]]

def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        return [img1,img2[-diff0:avg0,-diff1:avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]

    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]

def generate_face_correspondences(theImage1, theImage2):
    # Detect the points of face.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('libs/utils/shape_predictor_68_face_landmarks.dat')
    corresp = numpy.zeros((68,2))

    imgList = crop_image(theImage1,theImage2)
    list1 = []
    list2 = []
    j = 1

    for img in imgList:

        size = (img.shape[0],img.shape[1])
        if(j == 1):
            currList = list1
        else:
            currList = list2

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        dets = detector(img, 1)

        try:
            if len(dets) == 0:
                raise NoFaceFound
        except NoFaceFound:
            print("Sorry, but I couldn't find a face in the image.")

        j=j+1

        for k, rect in enumerate(dets):
            
            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)
            # corresp = face_utils.shape_to_np(shape)
            
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y
                # cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

            # Add back the background
            currList.append((1,1))
            currList.append((size[1]-1,1))
            currList.append(((size[1]-1)//2,1))
            currList.append((1,size[0]-1))
            currList.append((1,(size[0]-1)//2))
            currList.append(((size[1]-1)//2,size[0]-1))
            currList.append((size[1]-1,size[0]-1))
            currList.append(((size[1]-1),(size[0]-1)//2))

    # Add back the background
    narray = corresp/2
    narray = numpy.append(narray,[[1,1]],axis=0)
    narray = numpy.append(narray,[[size[1]-1,1]],axis=0)
    narray = numpy.append(narray,[[(size[1]-1)//2,1]],axis=0)
    narray = numpy.append(narray,[[1,size[0]-1]],axis=0)
    narray = numpy.append(narray,[[1,(size[0]-1)//2]],axis=0)
    narray = numpy.append(narray,[[(size[1]-1)//2,size[0]-1]],axis=0)
    narray = numpy.append(narray,[[size[1]-1,size[0]-1]],axis=0)
    narray = numpy.append(narray,[[(size[1]-1),(size[0]-1)//2]],axis=0)
    
    return [size,imgList[0],imgList[1],list1,list2,narray]


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(numpy.float32(srcTri), numpy.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(numpy.float32([t1]))
    r2 = cv2.boundingRect(numpy.float32([t2]))
    r = cv2.boundingRect(numpy.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = numpy.zeros((r[3], r[2], 3), dtype = numpy.float32)
    cv2.fillConvexPoly(mask, numpy.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


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

'''
██████   █████  ██████  ███████ ███████ ██████  
██   ██ ██   ██ ██   ██ ██      ██      ██   ██ 
██████  ███████ ██████  ███████ █████   ██████  
██      ██   ██ ██   ██      ██ ██      ██   ██ 
██      ██   ██ ██   ██ ███████ ███████ ██   ██ 
'''

def VideoMorph(Options):

	FPS = Options.FPS;
	Morph = Options.Morph;
	
	assert os.path.exists(Options.Sb1), "File not found " + Options.Sb1;
	assert os.path.exists(Options.Sb2), "File not found " + Options.Sb2;
	
	aligned_face_path = "./temp_0.png";	
	AlignFace(Options.Sb1, aligned_face_path);
	Options.Sb1 = aligned_face_path;
	aligned_face_path = "./temp_1.png";
	AlignFace(Options.Sb2, aligned_face_path);
	Options.Sb2 = aligned_face_path;		
	
	img1 = cv2.imread(Options.Sb1);
	img2 = cv2.imread(Options.Sb2);

	[size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)

	tri_list = make_delaunay(size[1], size[0], list3, img1, img2)

	Frames = Options.Length * FPS;
	p = subprocess.Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(FPS),'-s',str(size[1])+'x'+str(size[0]), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', Morph], stdin = subprocess.PIPE);

	# Convert Mat to float data type
	img1 = numpy.float32(img1);
	img2 = numpy.float32(img2);	
	
	for j in range(Frames):
		# Read array of corresponding points
		points = [];
		alpha = j/(Frames-1);
		# Compute weighted average point coordinates
		for i in range(len(points1)):
			x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
			y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
			points.append((x,y))
      # Allocate space for final output
			morphed_frame = numpy.zeros(img1.shape, dtype = img1.dtype)
		#
		for i in range(len(tri_list)):    
			x = tri_list[i][0];
			y = tri_list[i][1];
			z = tri_list[i][2];
			t1 = [points1[x], points1[y], points1[z]]
			t2 = [points2[x], points2[y], points2[z]]
			t = [points[x], points[y], points[z]]
      # Morph one triangle at a time.
			morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)
			#
			pt1 = (int(t[0][0]), int(t[0][1]))
			pt2 = (int(t[1][0]), int(t[1][1]))
			pt3 = (int(t[2][0]), int(t[2][1]))
			#
			cv2.line(morphed_frame, pt1, pt2, (255, 255, 255), 1, 8, 0)
			cv2.line(morphed_frame, pt2, pt3, (255, 255, 255), 1, 8, 0)
			cv2.line(morphed_frame, pt3, pt1, (255, 255, 255), 1, 8, 0)
		#
		res = Image.fromarray(cv2.cvtColor(numpy.uint8(morphed_frame), cv2.COLOR_BGR2RGB))
		res.save(p.stdin,'JPEG')

	p.stdin.close()
	p.wait()


def AlignFace(File, TempFile):
	print("Alignining %s" %(File));
	face_landmarks = landmarks_detector.get_landmarks(File);
	for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(File), start = 1):
		image_align(File, TempFile, face_landmarks)

'''
███████  █████   ██████ ███████     ███    ███  ██████  ██████  ██████  ██   ██ ███████ ██████  
██      ██   ██ ██      ██          ████  ████ ██    ██ ██   ██ ██   ██ ██   ██ ██      ██   ██ 
█████   ███████ ██      █████       ██ ████ ██ ██    ██ ██████  ██████  ███████ █████   ██████  
██      ██   ██ ██      ██          ██  ██  ██ ██    ██ ██   ██ ██      ██   ██ ██      ██   ██ 
██      ██   ██  ██████ ███████     ██      ██  ██████  ██   ██ ██      ██   ██ ███████ ██   ██ 
'''

# Delaunay MorphFace
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

	img1 = cv2.imread(Options.Sb1);
	img2 = cv2.imread(Options.Sb2);

	print("\nFace correspondences ... \n");

	[size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2);

	print("\nDelaunay Triangulation ... \n");

	tri_list = make_delaunay(size[1], size[0], list3, img1, img2);

	img1 = numpy.float32(img1);
	img2 = numpy.float32(img2);
	points = [];
	print("\nComputing weighted average point coordinates ... \n");
	for i in range(len(points1)):
		x = (1 - alpha) * points1[i][0] + alpha * points2[i][0];
		y = (1 - alpha) * points1[i][1] + alpha * points2[i][1];
		points.append((x,y))
	
	print("\nComputing Morph ... \n");
	morphed_frame = numpy.zeros(img1.shape, dtype = img1.dtype);
	#
	for i in range(len(tri_list)):    
		x = tri_list[i][0];
		y = tri_list[i][1];
		z = tri_list[i][2];
		t1 = [points1[x], points1[y], points1[z]];
		t2 = [points2[x], points2[y], points2[z]];
		t = [points[x], points[y], points[z]];
		morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha);
		#
	#
	morphed_frame = Image.fromarray(cv2.cvtColor(numpy.uint8(morphed_frame), cv2.COLOR_BGR2RGB));
	print("\nWritting morphed face : %s ... \n" %(Options.Morph));	
	morphed_frame.save(Options.Morph, "PNG");

	os.system("rm %s" %(Options.Sb1));
	os.system("rm %s" %(Options.Sb2));

def morph_2_faces_process(file1_path, file2_path, alpha, Morph_Results, temp_dir_path, log = False): #Sun 26 Feb 2026 11:21:45 GMT by MAPA
    # ---- Face align process
    # -- Align face 1
    aligned_face_path_file1 = temp_dir_path + "/temp_" + file1_path.split("/")[-1]
    original_file1_path = file1_path

    # aligned temp file existance validation
    if aligned_face_path_file1.split("/")[-1] not in os.listdir(temp_dir_path):
        if log: print("-- New temp file: ", aligned_face_path_file1)
        AlignFace(file1_path, aligned_face_path_file1);
    file1_path = aligned_face_path_file1;

    # -- Align face 2
    aligned_face_path_file2 = temp_dir_path + "/temp_" + file2_path.split("/")[-1]
    original_file2_path = file2_path

    # aligned temp file existance validation
    if aligned_face_path_file2.split("/")[-1] not in os.listdir(temp_dir_path):
        if log: print("-- New temp file: ", aligned_face_path_file2)
        AlignFace(file2_path, aligned_face_path_file2);
    file2_path = aligned_face_path_file2;


    # Read images
    img1 = cv2.imread(file1_path);
    img2 = cv2.imread(file2_path);

    # Generate correspondances
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2);

    # Delaunay
    tri_list = make_delaunay(size[1], size[0], list3, img1, img2);

    img1 = numpy.float32(img1);
    img2 = numpy.float32(img2);
    points = [];

    if log: print("\nComputing weighted average point coordinates ... \n");

    for i in range(len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0];
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1];
        points.append((x,y))

    if log: print("\nComputing Morph ... \n");
    
    morphed_frame = numpy.zeros(img1.shape, dtype = img1.dtype);
    #
    for i in range(len(tri_list)):    
        x = tri_list[i][0];
        y = tri_list[i][1];
        z = tri_list[i][2];
        t1 = [points1[x], points1[y], points1[z]];
        t2 = [points2[x], points2[y], points2[z]];
        t = [points[x], points[y], points[z]];
        morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha);
        #
    #
    morphed_frame = Image.fromarray(cv2.cvtColor(numpy.uint8(morphed_frame), cv2.COLOR_BGR2RGB));

    file_Morph_Results = Morph_Results + "/morph_" + original_file1_path.split("/")[-1].split(".")[-2] + "_" + original_file2_path.split("/")[-1].split(".")[-2] + ".png"

    if log: print("\nWritting morphed face : %s ... \n" %(file_Morph_Results));	

    # Save morphed face
    morphed_frame.save(file_Morph_Results, "PNG");

def Dir_Automation_MorphFace(Options): #Sun 26 Feb 2026 11:21:45 GMT by MAPA
    
    alpha = Options.Alpha;
    DirProportion = Options.DirProportion;

    # Parameter validation
    assert alpha > 0 and alpha < 1, "alpha not in [0,1]";
    assert DirProportion > 0 and DirProportion <= 1, "DirProportion not in [0,1]";
    assert os.path.exists(Options.ImageDir), "Directory not found " + Options.ImageDir;
    assert os.path.exists(Options.MorphDir), "Ouput Directory not found " + Options.MorphDir

    # Read directory
    all_files_path = sorted(os.listdir(Options.ImageDir))

    # Check temp dir existance
    temp_dir_path = "TempImages"
    os.makedirs(temp_dir_path, exist_ok=True)

    # Optimized and automated image morphing all vs all 
    for file_x_idx in range(int(len(all_files_path)*DirProportion)):
        for file_y_idx in range(file_x_idx, int(len(all_files_path)*DirProportion)):
            if file_y_idx != file_x_idx:   
                file1 = Options.ImageDir + "/" + all_files_path[file_x_idx] 
                file2 = Options.ImageDir + "/" + all_files_path[file_y_idx]

                # Process 2 faces
                morph_2_faces_process(file1, file2, alpha, Options.MorphDir, temp_dir_path, log = False)
    
    # Remove temp_dir
    os.system("rm -r %s" %(temp_dir_path));
