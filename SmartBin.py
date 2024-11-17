import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from time import sleep
from gpiozero import MotionSensor
from signal import pause
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

# # Set the GPIO pin number you've connected to the signal pin of the servo
# servo_pin = 17

# # Set the PWM frequency (pulse width modulation)
# PWM_frequency = 50  # Hertz

# # Setup PWM on the servo pin
# GPIO.setup(servo_pin, GPIO.OUT)
# pwm = GPIO.PWM(servo_pin, PWM_frequency)




def servo_rotate(inp):

    GPIO.setmode(GPIO.BCM)

    servo_pin = 17

    PWM_frequency = 50  # Hertz

    GPIO.setup(servo_pin, GPIO.OUT)
    pwm = GPIO.PWM(servo_pin, PWM_frequency)
    
    def angle_to_duty_cycle(angle):
        duty_cycle = (0.05 * PWM_frequency) + (0.19 * PWM_frequency * angle / 180)
        return duty_cycle

    def rotate(angle):
        duty_cycle = angle_to_duty_cycle(angle)
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Adjust this value as needed
        pwm.ChangeDutyCycle(0)

    pwm.start(0)
    
    choice = inp
    
    if choice == 'l':
        for angle in range(90, 20, -10):
            rotate(angle)
        sleep(2)
        for angle in range(20, 100, 10):
            rotate(angle)
    elif choice == 'r':
        for angle in range(90, 160, 10):
            rotate(angle)
        sleep(2)
        for angle in range(160, 80, -10):
            rotate(angle)
    else:
        rotate(90)
      
    # Cleanup
    pwm.stop()
    GPIO.cleanup()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 2)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

model = ResNet()

state_dict = torch.load('/home/pi/Desktop/smartbin/pytorch_state_dict.pth', map_location=torch.device('cpu'))

model.load_state_dict(state_dict)

model.eval()

class_labels = ['bio', 'non_bio']

model_new = ResNet()

state_dict_new = torch.load('/home/pi/Desktop/smartbin/new_data_model_1.pth', map_location=torch.device('cpu'))

model_new.load_state_dict(state_dict_new)

model_new.eval()

def classify_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        output_new = model_new(image)

    probabilities = F.softmax(output, dim=1)
    probabilities_list = probabilities[0].tolist()
    probabilities_new = F.softmax(output_new, dim=1)
    probabilities_new_list = probabilities_new[0].tolist()
    print(probabilities_list)
    print(probabilities_new_list)
    final_probabilities = [ (a + b) / 2 for a, b in zip(probabilities_list, probabilities_new_list) ]
    print(final_probabilities)
    predicted_class = final_probabilities.index(max(final_probabilities))
    if class_labels[predicted_class] == 'bio':
        servo_rotate('l')
    elif class_labels[predicted_class] == 'non_bio':
        servo_rotate('r')
    else:
        servo_rotate('n')
    return class_labels[predicted_class]


def motion_function():
    frame_count = 0
    max_frames = 20
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    sleep(2)
    while True:
        check, frame = webcam.read()
        print(check)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            webcam.release()  
            cv2.destroyAllWindows()
            break
        
        frame_count += 1
        if frame_count == max_frames:
            print("Capturing image...")
            print("Classifying image...")
            result = classify_image(frame)
            print("Result:", result)
            webcam.release()  
            cv2.destroyAllWindows()
            return 0      
        



print("PIR Motion Sensor Test (CTRL+C to exit)")
while True:


    GPIO. setmode (GPIO. BCM) 
    GPIO. setwarnings (False) 
    TRIG = 23 
    ECHO = 24 
    print ("Distance Measurement In Progress")
    GPIO. setup (TRIG, GPIO. OUT) 
    GPIO. setup (ECHO, GPIO. IN) 
    GPIO. output (TRIG, False) 
    print ("Waiting For Sensor To Settle")
    time. sleep (2) 
    GPIO. output (TRIG, True) 
    time. sleep (0.00001) 
    GPIO. output (TRIG, False) 
    while GPIO.input (ECHO) ==0: 
        pulse_start = time.time () 
    while GPIO.input (ECHO) ==1: 
        pulse_end = time.time () 
    pulse_duration = pulse_end - pulse_start 
    distance = pulse_duration * 17150 
    distance = round (distance, 2) 
    print ("Distance:", distance, "cm")
    if distance < 20.0:
        continue

    GPIO.setmode(GPIO.BCM)

    pir_pin = 27

    GPIO.setup(pir_pin, GPIO.IN)

    if GPIO.input(pir_pin):
        print("No Motion")
    else:
        print("Motion detected")
        motion_function()
    time.sleep(2)



