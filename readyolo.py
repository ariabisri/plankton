import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

model = YOLO('model/yolov8l-cls.pt')

# def main ():
#     yolo_pred('anjing.jpg')

def load_image(image_path):
    # Read the image
    img = image_path
    # img = cv2.imread(image_path)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 224x224 (the input size expected by YOLOv8 classification models)
    img = cv2.resize(img, (224, 224))
    # Convert to tensor and normalize
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    return img

def display_results(image_path, prediction):
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Get the predicted class and confidence
    label, confidence = prediction
    # Display the image with the predicted label and confidence
    plt.imshow(img)
    plt.title(f'Prediction: {label}, Confidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()



# Main function
def yolo_pred(image_path):
    # Load and preprocess the image
    image = load_image(image_path)
    
    # Run the model to get predictions
    results = model(image)

    name_list = results[0].names
    prob_list = results[0].probs.top5
    conf_list = results[0].probs.top5conf.cpu().tolist()

    top5name = []
    top5conf = []
    if len(prob_list)>3:
        top=3
    else:
        top = len(prob_list)
    for i in range(top):
        top5name.append(name_list[prob_list[i]])
        top5conf.append(f"{conf_list[i] * 100:.2f}%")   
    
   

    # output = results[0].probs.top5conf.cpu()[0] 
    # results[0].names[results[0].probs.top5[0]]
    # results[0].probs.top5[0][str(results[0].names)]
    # print (top5name)
    # print (top5conf)
    
    # confidence = results[0].probs.top1conf.cpu()

    # prediction_result ={'label':label, 'confidence':confidence}


    return top5name, top5conf
    
    # Display results
    # display_results(image_path, (label, confidence))

# Example usage
# image_path = 'path/to/your/image.jpg'  # Provide the path to your image here
# main(image_path)

# if __name__ == "__main__":
#     main()