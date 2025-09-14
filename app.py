from flask import Flask, request
from PIL import Image
import torch
import torchvision.transforms as transforms
from cnn_project import SimpleCNN, classes

app = Flask(__name__)

model = SimpleCNN()
model.load_state_dict(torch.load("cnn_cifar10.pth", map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output,1)
        return f"Predicted Class: {classes[predicted.item()]}"
    return '''
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
