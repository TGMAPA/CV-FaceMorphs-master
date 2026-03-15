from deepface import DeepFace

result = DeepFace.verify(
  img1_path = "0161_01.png",
  img2_path = "0146_01.png",
)

print(result)

objs = DeepFace.analyze(
  img_path = "0161_01.png", 
  actions = ['age', 'gender', 'race'],
)

print(objs)
