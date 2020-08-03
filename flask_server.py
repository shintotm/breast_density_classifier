import torch
import imageio
import numpy as np
import uuid
import flask

import models_torch as models
import utils


app = flask.Flask(__name__)

device = torch.device('cpu')

def load_image(image_path):
    image = imageio.imread(image_path)
    
    print(image.shape)
    h, w = image.shape[0], image.shape[1]

       
    image = image.astype(np.float32)
    print(image.min(), image.max())
    image -= np.mean(image)
    image /= np.std(image)
    
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image


def histo(img):
    parameters = {
        'bins_histogram': 50
    }
    model = models.BaselineHistogramModel(num_bins=50).to(device)
    model.load_state_dict(torch.load("saved_models/BreastDensity_BaselineHistogramModel/model.p"))
    x = torch.Tensor(utils.histogram_features_generator([
            img, img, img, img
        ], parameters)).to(device)


    # run prediction
    with torch.no_grad():
        prediction_density = model(x).cpu().numpy()
        return prediction_density

def get_predictions():
    data = {"success": False}
    task_id = str(uuid.uuid4())
    data ["id"] = task_id
    
    try:
        file = flask.request.files["image"]
        print("Received file: {}".format(file.filename))

        np_img = load_image(file.stream)

        prediction_density = histo(np_img)
        data['breast_density'] = str(np.argmax(prediction_density[0]) + 1)
        results = {
            'Almost entirely fatty (1):': str(prediction_density[0, 0]),
            'Scattered areas of fibroglandular density (2)': str(prediction_density[0, 1]),
            'Heterogeneously dense (3)': str(prediction_density[0, 2]),
            'Extremely dense (4)': str(prediction_density[0, 3])
            }
        print(results)
        data['predictions'] = results
        data['success'] = True
    
    except ValueError as e:
        data['Error'] = str(e)
    except Exception as e:
        data['Error'] = str(e) 
        
    if "Error" in data.keys():
        #logger.error("{} {}".format(task_id, data['Error']))
        print("{} {}".format(task_id, data['Error']))
    
    response = flask.jsonify(data)
    return response
    
    

@app.route("/api/v0", methods=["POST"])
def process_api():
    """
    cURL usage
    curl -F "image=@0_L_CC.png" {ipaddress}:5005/api/v0
    """
    return get_predictions()


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5005)
