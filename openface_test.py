import openface
import cv2

dlibFacePredictor = "openface/models/dlib/shape_predictor_68_face_landmarks.dat"


networkModel = "openface/models/openface/nn4.small2.v1.t7"
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, 96)

imgPath = "sample_to_rank.png"
bgrImg = cv2.imread(imgPath)
rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

bb = align.getLargestFaceBoundingBox(rgbImg)
alignedFace = align.align(96, rgbImg, bb,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
rep = net.forward(alignedFace)